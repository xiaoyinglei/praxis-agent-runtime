
import copy, importlib.util, json, sys
spec = importlib.util.spec_from_file_location("ledger", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
a = [
 {"tenant":"b","id":"x","seq":2,"amount":7,"extra":"keep"},
 {"tenant":"a","id":"x","seq":3,"amount":2},
 {"tenant":"a","id":"x","seq":1,"amount":900},
 {"tenant":"a","id":"x","seq":3,"amount":-2,"extra":"later tie"},
 {"tenant":"a","id":"y","seq":2,"deleted":True,"amount":999},
 {"tenant":"a","id":"y","seq":1,"amount":500},
 {"tenant":"c","id":"z","seq":1,"deleted":True},
 {"tenant":"c","id":"z","seq":2,"amount":0},
 {"tenant":"b","id":"large","seq":1,"amount":9007199254740993},
 {"tenant":"d","id":"gone","seq":1,"deleted":True},
 {"tenant":"c","id":"missing","seq":1},
]
original=copy.deepcopy(a)
expected=[a[3],a[8],a[0],a[10],a[7]]
assert m.reconcile(a)==expected, (m.reconcile(a),expected)
assert a==original and m.reconcile([])==[]
if sys.argv[2]=="2":
 assert m.totals(a)=={"a":-2,"b":9007199254741000,"c":0}, m.totals(a)
 assert m.totals([])=={} and a==original
print(json.dumps({"pass":True,"phase":int(sys.argv[2])}))
