import claripy

def convert(constraints):
    z = claripy.backend_manager.backends.smtlib_z3
    zs = z.solver()
    z.add(zs, constraints)

    variables, constraints = z._get_all_vars_and_constraints(solver=zs)
    smt_script = z._get_satisfiability_smt_script(constraints=constraints, variables=variables)
    zs.p.kill()
    zs.p.wait()
    return smt_script

if __name__ == '__main__':
    a = claripy.BVS('a', 64)
    print(convert([a > 10, a < 500]))
