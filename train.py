from param_parser import parameter_parser
from solver import Solver


def main():
    args = parameter_parser()
    solver = Solver(args)
    solver.init_training()
    solver.train()
    solver.test()


if __name__ == '__main__':
    main()