#!/usr/bin/env python3

import angr
import claripy
import logging
import sys
import random

import claripy_searchmc
import interpret

if len(sys.argv) < 3:
    print("Usage: python analyze.py binary rounds [-v]")

p = angr.Project(sys.argv[1], exclude_sim_procedures_list=['srand', 'rand'])
rounds = int(sys.argv[2])

if '-v' in sys.argv:
    logging.getLogger('angr.engines.engine').setLevel("INFO")
logging.getLogger('angr.state_plugins.symbolic_memory').setLevel('ERROR')

class SymbolicPid(angr.SimProcedure):
    def run(self): # pylint: disable=arguments-differ
        pid = self.state.solver.BVS('pid', 15, key=('entropy', 'getpid')).zero_extend(64-15)
        self.state.solver.add(pid == random.randrange(0, 2**15))
        return pid

class TrueRand(angr.SimProcedure):
    def run(self): # pylint: disable=arguments-differ
        val = self.state.solver.BVS('true_random', 32, key=('entropy', 'true_rand')).zero_extend(64-32)
        self.state.solver.add(val == random.randrange(0, 2**32))
        return val

p.hook_symbol('getpid', SymbolicPid())
p.hook_symbol('true_rand', TrueRand())

def single_path():
    simgr = p.factory.simulation_manager()
    simgr.run()

    if not simgr.deadended:
        print("OH NO: couldn't find a deadended state")
        print("Errored states:")
        for e in simgr.errored:
            print(e)
        sys.exit(1)

    state = simgr.deadended[0]

    content = state.solver.Concat(*[entry[0] for entry in state.posix.stdout.content])
    return content

def analyze_mine(content):
    bitmap = interpret.interpret(content)
    allbits = interpret.mix_bits(bitmap)
    collapsed = interpret.simplify(allbits)
    return collapsed

def analyze_searchmc(content):
    thresh = float(len(content)) - 1
    var = claripy.BVS('TARGET', len(content))
    con = [var == content]
    import time
    while True:
        t0 = time.time()
        print('thresh', thresh)
        low, high = claripy_searchmc.mc(var, con, 1, thresh)
        t1 = time.time()
        if low == high or t1 - t0 > 10:
            break

    return low, high

data = []
for i in range(1, rounds + 1):
    output = single_path()
    bits = analyze_mine(output)
    datum = interpret.compute_final(bits)
    data.append(datum)

    #xbits_low, xbits_high = analyze_searchmc(output)
    print('Round %d/%d:' % (i, rounds), datum, 'bits of entropy')
    #print('searchmc:', xbits_low, xbits_high)

print('')
print(sys.argv[1])
print('min entropy:', min(data), 'bits')
print('max entropy:', max(data), 'bits')
print('')
