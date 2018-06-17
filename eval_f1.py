from projects.convai2.eval_f1 import setup_args, eval_f1

if __name__ == '__main__':
    parser = setup_args()
    parser.set_params(
        model='projects.convai2.convai.t_agent:DSSMAgent'
    )
    opt = parser.parse_args()
    eval_f1(opt, print_parser=parser)
