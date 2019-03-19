#!/usr/bin/env python3

import angr
import claripy
import logging
import sys
import random

import claripy_searchmc
import interpret

l = logging.getLogger('symtropy.analyze')

if len(sys.argv) < 3:
    print("Usage: python analyze.py binary rounds [-v] [-- arguments]")

logging.getLogger('symtropy').setLevel('WARNING')
if '-v' in sys.argv:
    logging.getLogger('angr.engines.engine').setLevel("INFO")
    logging.getLogger('symtropy').setLevel('INFO')
logging.getLogger('angr.state_plugins.symbolic_memory').setLevel('ERROR')

class TrueRand(angr.SimProcedure):
    def run(self, name=None, bits=None, reuse=True): # pylint: disable=arguments-differ
        key = ('entropy', name)
        if key in self.state.globals:
            val = self.state.globals[key]
        else:
            val = self.state.solver.BVS(name, bits, key=key)
            self.state.solver.add(val == random.randrange(0, 2**bits))

            if reuse:
                self.state.globals[key] = val

        return val.zero_extend(self.arch.bits-bits)

def load_project():
    p = angr.Project(sys.argv[1], exclude_sim_procedures_list=['srand', 'rand', 'memset', 'malloc', 'free', 'realloc', 'bzero', 'memcpy', 'strlen', 'strcpy', 'strncpy', 'strcmp', 'strncmp'])

    p.hook_symbol('getpid', TrueRand(name='getpid', bits=15))
    p.hook_symbol('true_rand', TrueRand(name='true_rand', bits=32, reuse=False))
    return p

def single_path(p):
    try:
        args = [sys.argv[1]] + sys.argv[sys.argv.index('--')+1:]
    except ValueError:
        args = [sys.argv[1]]

    initial_state = p.factory.full_init_state(args=args, add_options=angr.options.unicorn | {angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY}, remove_options={angr.options.ALL_FILES_EXIST}, concrete_fs=True)
    simgr = p.factory.simulation_manager(initial_state)
    simgr.run(until=lambda lsm: len(lsm.active) != 1)

    if not simgr.deadended:
        print("OH NO: couldn't find a deadended state")
        if simgr.errored:
            print("Errored states:")
            for e in simgr.errored:
                print(e)
        elif simgr.active:
            print("Active states:")
            for s in simgr.active:
                print(s)
        else:
            print("Unknown result??????")
            print(simgr)
        import ipdb; ipdb.set_trace()
        sys.exit(1)

    state = simgr.deadended[0]

    content = state.solver.Concat(*[entry[0] for entry in state.posix.stdout.content])
    l.info("Single-path finished")
    return content

def analyze_mine(content):
    bitmap = interpret.interpret(content)
    allbits = interpret.mix_bits(interpret.convert_to_bools(interpret.bake_masks(bitmap)))
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

def main():
    p = load_project()
    rounds = int(sys.argv[2])
    data = []
    for i in range(1, rounds + 1):
        output = single_path(p)
        bits_desc = analyze_mine(output)
        bits_num = interpret.compute_final(bits_desc)
        data.append(bits_num)

        #xbits_low, xbits_high = analyze_searchmc(output)
        print('Round %d/%d:' % (i, rounds), bits_num, 'bits of entropy')
        #print('searchmc:', xbits_low, xbits_high)

    print('')
    print(sys.argv[1])
    print('min entropy:', min(data), 'bits')
    print('max entropy:', max(data), 'bits')
    print('')

main()
