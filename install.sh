#!/bin/sh
# Praxis independent installer. No Python/package manager is needed on the host.
set -eu
umask 077

fail() { printf '\n安装未完成：%s\n' "$*" >&2; exit 1; }
quote() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }
repository=https://github.com/xiaoyinglei/praxis-agent-runtime
install_root=${PRAXIS_INSTALL_DIR:-"$HOME/.local/share/praxis"}
bin_dir=${PRAXIS_BIN_DIR:-"$HOME/.local/bin"}
case "$install_root:$bin_dir" in *'
'*) fail '安装目录不能包含换行。';; esac
case "$install_root" in /*) ;; *) fail 'PRAXIS_INSTALL_DIR 必须是绝对路径。';; esac
case "$bin_dir" in /*) ;; *) fail 'PRAXIS_BIN_DIR 必须是绝对路径。';; esac
for utility in curl tar awk sed mktemp; do
    command -v "$utility" >/dev/null 2>&1 || fail "缺少系统命令：$utility"
done
case "$(uname -s)" in Darwin) system=darwin;; Linux) system=linux;; *) fail '当前支持 macOS 和 Linux。';; esac
case "$(uname -m)" in arm64|aarch64) arch=arm64;; x86_64|amd64) arch=x86_64;; *) fail '当前支持 arm64 和 x86_64。';; esac
if command -v shasum >/dev/null 2>&1; then
    hash_command=shasum
elif command -v sha256sum >/dev/null 2>&1; then
    hash_command=sha256sum
else
    fail '系统缺少 shasum 或 sha256sum，无法核对下载文件。'
fi
if [ "$system" = linux ] && ! getconf GNU_LIBC_VERSION >/dev/null 2>&1; then
    fail '独立发行包需要 glibc Linux；Alpine 等 musl 系统请使用 Python 安装方式。'
fi
launcher=$bin_dir/agent
if [ -e "$launcher" ] || [ -L "$launcher" ]; then
    [ ! -L "$launcher" ] && [ -f "$launcher" ] &&
        sed -n '2p' "$launcher" | awk '$0 == "# Praxis managed launcher" { found=1 } END { exit !found }' ||
        fail "$launcher 已由其它方式安装。请先卸载原安装，或通过 PRAXIS_BIN_DIR 选择另一个目录。"
fi
printf 'Praxis 安装 · %s %s\n' "$system" "$arch"
printf '1/3 下载独立发行包\n'
if [ -n "${PRAXIS_VERSION:-}" ]; then
    tag=$PRAXIS_VERSION
else
    release=$(curl -fsSL --retry 2 --connect-timeout 15 -o /dev/null -w '%{url_effective}' "$repository/releases/latest") ||
        fail '无法获取发行版本，请检查 GitHub 网络连接。'
    tag=${release##*/}
fi
case "$tag" in ''|*[!a-zA-Z0-9._-]*) fail '发行版本格式不正确。';; esac
asset=praxis-$system-$arch.tar.gz
base=$repository/releases/download/$tag
work=$(mktemp -d "${TMPDIR:-/tmp}/praxis-download.XXXXXX")
stage=
locked=
new_launcher=
cleanup() {
    rm -rf "$work"
    [ -z "$stage" ] || rm -rf "$stage"
    [ -z "$new_launcher" ] || rm -f "$new_launcher"
    [ -z "$locked" ] || rmdir "$install_root/.install-lock" 2>/dev/null || :
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
curl -fsSL --retry 2 --connect-timeout 15 "$base/$asset.sha256" -o "$work/checksum" ||
    fail "发行版 $tag 尚无 $system-$arch 安装包，或下载连接失败。"
curl -fL --retry 2 --connect-timeout 15 --progress-bar "$base/$asset" -o "$work/$asset" ||
    fail '下载中断，重新运行同一条安装命令即可。'
expected=$(awk -v name="$asset" '$2 == name { print $1 }' "$work/checksum")
[ "${#expected}" -eq 64 ] || fail '发行文件校验信息不完整。'
case "$expected" in *[!a-f0-9]*) fail '发行文件校验信息无效。';; esac
if [ "$hash_command" = shasum ]; then
    actual=$(shasum -a 256 "$work/$asset" | awk '{print $1}')
else
    actual=$(sha256sum "$work/$asset" | awk '{print $1}')
fi
[ "$actual" = "$expected" ] || fail '下载文件的 SHA-256 不一致，未安装。'
printf '2/3 校验并安装\n'
mkdir -p "$install_root/versions" "$bin_dir"
mkdir "$install_root/.install-lock" 2>/dev/null || fail '另一个安装进程正在运行。'
locked=yes
stage=$(mktemp -d "$install_root/versions/.install.XXXXXX")
tar -tzf "$work/$asset" | awk '
    $0 != "praxis/" && $0 !~ /^praxis\// { bad=1 }
    /(^|\/)\.\.(\/|$)/ { bad=1 }
    END { exit bad }
' || fail '发行包目录结构不正确。'
tar -xzf "$work/$asset" --strip-components 1 -C "$stage" || fail '解包失败；旧安装保持可用。'
[ -x "$stage/python/bin/python3" ] && [ -x "$stage/bin/agent" ] && [ -f "$stage/VERSION" ] ||
    fail '发行包缺少运行文件。'
version=$(cat "$stage/VERSION")
case "$version" in ''|*[!a-zA-Z0-9._-]*) fail '安装包版本格式不正确。';; esac
destination=$install_root/versions/$version-$(printf '%s' "$expected" | cut -c 1-12)
if [ -e "$destination" ]; then
    [ -f "$destination/INSTALL_SHA256" ] && [ "$(cat "$destination/INSTALL_SHA256")" = "$expected" ] ||
        fail "版本目录已存在且不是本安装器管理的内容：$destination"
else
    printf '%s\n' "$expected" > "$stage/INSTALL_SHA256"
    mv "$stage" "$destination"
    stage=
fi
# Recheck ownership under the installation lock before publishing the launcher.
if [ -e "$launcher" ] || [ -L "$launcher" ]; then
    [ ! -L "$launcher" ] && [ -f "$launcher" ] &&
        [ "$(sed -n '2p' "$launcher")" = '# Praxis managed launcher' ] || fail '已有 agent 命令发生变化，未覆盖。'
fi
new_launcher=$(mktemp "$bin_dir/.praxis-agent.XXXXXX")
printf '#!/bin/sh\n# Praxis managed launcher\nexec %s "$@"\n' "$(quote "$destination/bin/agent")" > "$new_launcher"
chmod 755 "$new_launcher"
mv -f "$new_launcher" "$launcher"
new_launcher=
printf '3/3 完成 · Praxis %s\n' "$version"
case ":$PATH:" in
    *":$bin_dir:"*)
        if [ "$(command -v agent 2>/dev/null || :)" = "$launcher" ]; then
            command_name=agent
        else
            command_name=$(quote "$launcher")
            printf '当前终端有另一个同名命令优先；请使用下方完整路径。\n'
        fi
        ;;
    *)
        command_name=$(quote "$launcher")
        if [ "$bin_dir" = "$HOME/.local/bin" ]; then
            case "${SHELL:-}" in
                */zsh) rc=$HOME/.zshrc;;
                */bash) if [ "$system" = darwin ]; then rc=$HOME/.bash_profile; else rc=$HOME/.bashrc; fi;;
                *) rc=;;
            esac
            if [ -n "$rc" ]; then
                line='export PATH="$HOME/.local/bin:$PATH" # Praxis CLI'
                if ! [ -f "$rc" ] || ! grep -Fqx "$line" "$rc"; then
                    printf '\n%s\n' "$line" >> "$rc" || printf '无法更新 shell 配置，请使用下方完整路径。\n' >&2
                fi
                printf '已配置命令路径；重新打开终端后可直接运行 agent。\n'
            fi
        fi
        ;;
esac
printf '\n在你的项目目录中执行：\n  %s setup\n  %s chat\n' "$command_name" "$command_name"
printf '\n默认连接 DeepSeek，首次配置时输入你自己的 API Key。\n'
if [ "$system" = linux ] && { [ ! -x /usr/bin/bwrap ] || ! ldconfig -p 2>/dev/null | grep -q libseccomp.so.2; }; then
    printf '\nLinux 命令沙箱需要 Bubblewrap 和 libseccomp；请按项目 Ubuntu 部署说明准备系统后端。\n'
fi
