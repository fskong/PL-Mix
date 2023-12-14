def common_args(parser):
    ''' dataset '''
    parser.add_argument('--dataset', type=str, default='Review-small', choices=['Review-small', 'Review-large', 'Benchmark'], help='Dataset to run')
    parser.add_argument('-s', '--source', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('--data_dir', type=str, default='../data', help='The path to download dataset.')
    ''' optimization '''
    parser.add_argument('--model_name', default='/home/LAB/kongfs/PHD/UDA/Proto/bert-base-uncased', type=str)  # , choices=['bert-base-uncased', 'roberta-base', 'roberta-base-uncased']
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--domain-lr', type=float, default=2e-5)
    parser.add_argument('--weight-decay', type=float, default=0)
    ''' environment '''
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'], help='Selected device.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--timestamp', type=str, default=None, help='Experiment timestamp.')
    parser.add_argument('--workers', type=int, default=0)
    ''' logger '''
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--output_dir', type=str, default='saved_models', help='Path to save model and logs.')
