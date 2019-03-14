from uno_data import CombinedDataLoader, CombinedDataGenerator
import pandas as pd
import time

loader = CombinedDataLoader(seed=2018)
loader.load(cache='cache/CTRP',
            ncols=0,
            agg_dose=None,
            cell_features=['rnaseq'],
            drug_features=['descriptors','fingerprints'],
            drug_median_response_min=-1,
            drug_median_response_max=1,
            use_landmark_genes=True,
            use_filtered_genes=False,
            cell_feature_subset_path="",
            drug_feature_subset_path="",
            preprocess_rnaseq='combat',
            single=False,
            train_sources=['CTRP'],
            test_sources=['train'],
            embed_feature_source=False,
            encode_response_source=False,
            )
target = 'Growth'
val_split = 0.2
train_split = 1 - val_split

loader.partition_data(cv_folds=1, train_split=train_split, val_split=val_split,
                      cell_types=None, by_cell=None, by_drug=None,
                      cell_subset_path=None, drug_subset_path=None)

batch_size = 8192
train_gen = CombinedDataGenerator(loader, batch_size=batch_size, shuffle=False)
val_gen = CombinedDataGenerator(loader, partition='val', batch_size=batch_size, shuffle=False)

_dtype_ = 'float32'

store = pd.HDFStore('df_train.h5')
for i in range(train_gen.steps):
    s1 = time.time()
    x_train_list, y_train = train_gen.get_slice(size=batch_size, dataframe=True, single=False)
    s2 = time.time()
    # if i == 0:
    #    df_columns = pd.DataFrame(data=x_train.columns)
    #    store.put('x_columns', df_columns, format='table')

    for j,val in enumerate(x_train_list):
        val.columns = [''] * len(val.columns)
        store.append('x_train_%d' % j, val.astype(_dtype_), format='table', data_column=True)

    store.append('y_train', y_train['Growth'].astype(_dtype_), format='table', data_column=True)
    s3 = time.time()
    print("step: %d / %d slice: %.3f, store: %.3f, total: %.3f" % (i, train_gen.steps, (s2-s1), (s3-s2), (s3-s1)))

for i in range(val_gen.steps):
    s1 = time.time()
    x_val_list, y_val = val_gen.get_slice(size=batch_size, dataframe=True, single=False)
    s2 = time.time()

    for j,val in enumerate(x_val_list):
        val.columns = [''] * len(val.columns)
        store.append('x_val_%d' % j, val.astype(_dtype_), format='table', data_column=True)

    store.append('y_val', y_val['Growth'].astype(_dtype_), format='table', data_columns=True)
    s3 = time.time()
    print("step: %d / %d slice: %.3f, store: %.3f, total: %.3f" % (i, val_gen.steps, (s2-s1), (s3-s2), (s3-s1)))

store.close()
