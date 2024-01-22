
PREFIX_TASK_ID = "task"

CONFIGURATION = "configuration"

# nodes
SELECTED_CLIENTS = 'selected_nodes'
GROUPED_CLIENTS = 'grouped_nodes'

# communication cost
DOWNLOAD_SIZE = 'download_size'
TRAIN_DOWNLOAD_SIZE = 'train_download_size'
TRAIN_UPLOAD_SIZE = 'train_upload_size'
TEST_DOWNLOAD_SIZE = 'test_download_size'
TEST_UPLOAD_SIZE = 'test_upload_size'

# distribute time
UPLOAD_TIME = "upload_time"
TRAIN_UPLOAD_TIME = "train_upload_time"
TEST_UPLOAD_TIME = "test_upload_time"
TRAIN_DISTRIBUTE_TIME = "train_distribute_time"
TEST_DISTRIBUTE_TIME = "test_distribute_time"

# time
ROUND_TIME = "round_time"
TRAIN_TIME = 'train_time'
TEST_TIME = 'test_time'
TRAIN_EPOCH_TIME = 'train_epoch_time'

# performance
TRAIN_ACCURACY = 'train_accuracy'
TRAIN_LOSS = 'train_loss'
TRAIN_BEST_LOSS = "train_best_loss"
TRAIN_STATIC_REGRET = "train_static_regret"
AVG_TRAIN_LOSS = 'avg_train_loss'
AVG_TRAIN_ACCURACY = "avg_train_accuracy"
AVG_TRAIN_STATIC_REGRET = "avg_train_static_regret"


TEST_ACCURACY = 'test_accuracy'
TEST_LOSS = 'test_loss'
TEST_CONSENSUS_ERROR = "test_consensus_error"

TEST_AVG_MODEL_ACCURACY = 'test_avg_model_accuracy'
TEST_AVG_MODEL_LOSS = 'test_avg_model_loss'

CUMULATIVE_TEST_ROUND = "cumulative_test_round"
# CONSENSUS_ERROR = "consensus_error"
NODE_METRICS = "node_metrics"

# process
TEST_IN_SERVER = "test_in_controller"
TEST_IN_CLIENT = "test_in_node"
AGGREGATION_CONTENT_ALL = "all"
AGGREGATION_CONTENT_PARAMS = "parameters"
SAVED_MODELS = "saved_models"
MEAN = "Mean"

# train and test params
MODEL = "model"
GRADIENT_MESSAGE = "gradient_message"
GRADIENT = "gradient"
TEST_DATA_SIZE = "test_data_size"
TRAIN_DATA_SIZE = "train_data_size"

ACCURACY = "accuracy"
LOSS = "loss"

# general
EXTRA = "extra"  # for not specifically defined metrics
METRIC = "metric"
CPU = "cpu"
ONLINE_BEST_MODEL = "online_best_model"
LEARNING_RATE = "learning_rate"

