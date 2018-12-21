import os
import math
import claripy
import claripy_smt2
import subprocess

basedir = os.path.dirname(__file__)

def mc(var, constraints, confidence, threshold):
    assert var.op == 'BVS'
    if not any(var.args[0] in c.variables for c in constraints):
        return [float(len(var)), float(len(var))]

    smtscript = claripy_smt2.convert(constraints)
    with open(os.path.join(basedir, 'SearchMC', 'a.smt2'), 'w') as fp:
        fp.write(smtscript)

    # ./SearchMC.pl -cl=1 -thres=1 -input_type=smt -output_name=a_0_64 ./a.smt2
    proc = subprocess.Popen(['./SearchMC.pl', '-cl=%f' % confidence, '-thres=%f' % threshold, '-input_type=smt', '-output_name=%s' % var.args[0], './a.smt2'], cwd=os.path.join(basedir, 'SearchMC'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()

    for line in stdout.splitlines() + stderr.splitlines():
        if line.startswith(b'a.smt2.cnf'):
            return list(map(float, line.split()[1:3]))
        elif line.startswith(b'Output not found'):
            return [0., 0.]

    raise Exception("Failed to model-count", stdout)

def p(constraints, confidence, threshold, mode='high'):
    roots = [claripy.BVS(n, int(n.split('_')[-1]), explicit_name=True) for n in set().union(*[c.variables for c in constraints])]
    megaval = claripy.Concat(*roots)
    megavar = claripy.BVS('RESULT', len(megaval))
    mc_low, mc_high = mc(megavar, constraints + [megavar == megaval], confidence, threshold)
    if mode == 'high':
        count = mc_high
    elif mode == 'low':
        count = mc_low
    elif mode == 'avg':
        count = (mc_high + mc_low) / 2
    elif mode == 'test':
        print('high', math.exp(mc_high - len(megaval)))
        print('low', math.exp(mc_low - len(megaval)))
        print('avg', math.exp((mc_high + mc_low) / 2 - len(megaval)))
        count = mc_high
    else:
        raise ValueError(mode)

    return math.exp(count - len(megaval))

if __name__ == '__main__':
    a = claripy.BVS('a', 32)
    b = claripy.BVS('b', 16)
    #print(mc(a, [a == claripy.Concat(b ^ 0x1234, b ^ 0x5678), a < 10000000], 1, 1))
    print(p([a[0] == 1], 0.8, 0.8, mode='test'))
