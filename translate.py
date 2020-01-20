from param_parser import parameter_parser
from solver import Solver


def main():
    args = parameter_parser()
    solver = Solver(args)
    solver.translate('data/test.de-en.de')


if __name__ == '__main__':
    main()
