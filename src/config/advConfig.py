from config.common import common_args


def advParser(subparsers):
  parser = subparsers.add_parser('ADV')
  common_args(parser)
  parser.add_argument('--alpha', type=float, default=0.1)
  parser.add_argument('--mix_alpha_c', type=float, default=1)
  parser.add_argument('--mix_alpha_d', type=float, default=1)
  # parser.add_argument('--domain-lr', type=float, default=5e-5)
  parser.add_argument('--prompt-k', type=int, default=2)
  return parser

