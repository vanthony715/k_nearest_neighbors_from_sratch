run: python vasquez_project_2_main.py [--arguments]

Output will print in the terminal to indicate job information, process step, and evaluation metrics.

arguments include:

parser.add_argument('--data_folder_name', type = str ,default = '/data',
                    help='Name of the folder where the data and names files are located'),

parser.add_argument('--dataset_name', type = str ,default = '/abalone',
                    help='Name of the folder where the data and names files are located'),

parser.add_argument('--namespath', type = str ,default = 'data/abalone.names',
                    help='Path to dataset names'),

parser.add_argument('--discretize_data', type = bool ,default = False,
                    help='Should dataset be discretized?'),

parser.add_argument('--quantization_number', type = int ,default = 3,
                    help='If discretized, then quantization number'),

parser.add_argument('--standardize_data', type = bool , default = True,
                    help='Should data be standardized?'),

parser.add_argument('--k_folds', type = int , default = 5,
                    help='Number of folds for k-fold validation'),

parser.add_argument('--k_neighbors', type = int , default = 6,
                    help='Number of Kernals for KNN'),

parser.add_argument('--min_examples', type = int , default = 15,
                    help='Drop classes with less examples then this value'),

parser.add_argument('--remove_orig_cat_col', type = bool , default = True,
                    help='Remove the original categorical columns for data encoding'),

parser.add_argument('--fine_tune_k', type = bool , default = False,
                    help='fine tune knn model')

parser.add_argument('--knn_values', type = int , default = 10,
                    help='test values of knn up to this number')

parser.add_argument('--fine_tune_error', type = bool , default = True,
                    help='fine tune regression error')

parser.add_argument('--error_values', type = int , default = 10,
                    help='value to fine-tune over for regression')

parser.add_argument('--min_gain', type = float , default = 0.05,
                    help='minimum percentage for each k_neighbor')

parser.add_argument('--editted_knn', type = bool , default = False,
                    help='use edited knn on train/test set ? ')

parser.add_argument('--condensed_knn', type = bool , default = False,
                    help='use condensed knn ? ')

parser.add_argument('--max_error', type = float , default = 50,
                    help='max error for regression to add to drop list')

parser.add_argument('--allowed_unchanged_cnt', type = int , default = 50,
                    help='Number of iteration that the unchanged count can stagnate')