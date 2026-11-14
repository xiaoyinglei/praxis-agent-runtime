"""内容寻址 artifact 目录(设计 §5.7 第 1 层;D10)。

超 ``max_model_output_bytes`` 的工具结果在**写入时**外部化到这里:item payload
只留头部片段 + 指针文本 + ``artifact_digest``/``artifact_bytes``,全文只存
``<root>/<sha256[:2]>/<sha256>``(B22:全文只存一份)。

纪律:
- 不新增 DB 表——artifact 是纯文件目录,写事务仍只有 INSERT(items);
- 每次 ``get`` 读后重算 sha256,不符即 ``ArtifactIntegrityError``(内容寻址的
  完整性不是"信文件名",而是"信内容");
- ``put`` 用同目录 tmp + ``fsync`` + ``os.replace``,权限 0600;
- ``gc`` 是**离线**函数:任何写路径/turn 循环都不得调用它(测试只钉死 execute
  写路径;其余调用点必须自查)。
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

from praxis.protocol.artifacts import ArtifactIntegrityError
from praxis.protocol.items import Item

__all__ = ["ArtifactIntegrityError", "ArtifactStore"]

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_SHARD_RE = re.compile(r"^[0-9a-f]{2}$")


class ArtifactStore:
    """``root`` 即 artifact 目录(生产为 ``<workspace>/.praxis/artifacts``)。"""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    # -- 写入与读取 -------------------------------------------------------------

    def put(self, data: bytes) -> str:
        """把 ``data`` 落到 ``<root>/<digest[:2]>/<digest>``;返回 sha256 hex。

        同目录 tmp → fsync → chmod 0600 → os.replace:读侧永远看不到半写文件。
        内容寻址下同一内容重复 put 是幂等的(名字即内容摘要)。
        """
        digest = hashlib.sha256(data).hexdigest()
        shard = self._root / digest[:2]
        shard.mkdir(parents=True, exist_ok=True)
        target = shard / digest
        if not target.is_symlink() and target.is_file():
            return digest  # 内容寻址:同内容已存在,省一次全量写
        fd, tmp_name = tempfile.mkstemp(prefix=f".{digest}.", suffix=".tmp", dir=str(shard))
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, target)
            # rename 的目录项也要落盘:否则崩溃后 item 声称 blob 存在而文件丢失。
            dir_fd = os.open(shard, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return digest

    def get(self, digest: str) -> bytes | None:
        """读全文并重算 sha256;缺失返回 None,不符抛 ``ArtifactIntegrityError``。"""
        path = self._path(digest)
        try:
            data = path.read_bytes()
        except OSError:
            # 缺失/目录替代/权限错误一律按"不可读"处理,由工具映射为
            # artifact_not_found——畸形输入永远是结果,不是异常。
            return None
        if hashlib.sha256(data).hexdigest() != digest:
            raise ArtifactIntegrityError(
                f"artifact content does not match digest {digest}"
            )
        return data

    def size(self, digest: str) -> int | None:
        """artifact 字节数;缺失返回 None。"""
        path = self._path(digest)
        try:
            info = path.stat()
        except OSError:
            return None
        return info.st_size if stat.S_ISREG(info.st_mode) else None  # 目录/特殊文件不算 artifact

    def iter_digests(self) -> Iterator[str]:
        """枚举目录中形状合法的 digest(读路径 fail-open:坏目录跳过)。

        符号链接的分片目录/文件一律跳过:GC 是离线跑的,不能靠 ``.praxis``
        地板保证目录内没有"指向存储外"的链接。
        """
        if not self._root.is_dir():
            return
        for shard in sorted(self._root.iterdir()):
            if shard.is_symlink() or not shard.is_dir() or _SHARD_RE.fullmatch(shard.name) is None:
                continue
            for entry in sorted(shard.iterdir()):
                if entry.is_symlink() or not entry.is_file():
                    continue
                if _DIGEST_RE.fullmatch(entry.name) is not None:
                    yield entry.name

    # -- 离线 GC ----------------------------------------------------------------

    def gc(self, referenced: set[str], *, now: float, grace_seconds: float = 3600.0) -> int:
        """删除"未被引用且 mtime 早于宽限期"的 artifact;返回删除数。

        **离线调用**:不在任何写路径或 turn 循环里(测试只钉死 execute 写路径;
        其余调用点必须自查)。``referenced`` 必须是**完整跨 thread 的**引用集合,
        否则会删掉其他 thread 仍在引用的 blob。宽限期防止刚写入、尚未进入
        item 日志的 artifact 被并发 GC 误删。符号链接/非常规文件一律不碰。
        """
        removed = 0
        for digest in list(self.iter_digests()):
            if digest in referenced:
                continue
            if (self._root / digest[:2]).is_symlink():  # iter_digests 已过滤,纵深防御
                continue
            path = self._root / digest[:2] / digest
            try:
                info = path.lstat()
            except OSError:
                continue
            if not stat.S_ISREG(info.st_mode):  # lstat:symlink 不是常规文件
                continue
            if now - info.st_mtime <= grace_seconds:
                continue
            try:
                path.unlink()
            except OSError:
                continue
            removed += 1
        return removed

    @staticmethod
    def referenced_by(items: Sequence[Item]) -> set[str]:
        """扫描 ``tool_result`` payload 的 ``metadata["artifact_digest"]``。

        只从**交给它的** items 收集引用:调用方必须传完整跨 thread 的 item 日志
        (如逐 thread ``read_items`` 的并集),否则按单 turn/单 thread 喂进来会
        漏掉其他范围仍在引用的 blob,GC 将误删。未知 kind / 畸形 payload 一律
        fail-open 跳过(读路径永不因一条坏记录炸掉)。
        """
        found: set[str] = set()
        for item in items:
            if item.kind != "tool_result":
                continue
            metadata = item.payload.get("metadata")
            if not isinstance(metadata, Mapping):
                continue
            digest = metadata.get("artifact_digest")
            if isinstance(digest, str) and digest:
                found.add(digest)
        return found

    # -- 内部 -------------------------------------------------------------------

    def _path(self, digest: str) -> Path:
        if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
            raise ValueError("invalid artifact digest")
        return self._root / digest[:2] / digest
