import pandas as pd
from utils import FrameHandler, prepare_dataframe
from model import LogisticRegressionBinary, LogisticRegressionMultinomial

UNNESSESARY_FEATURES = ['Defense Against the Dark Arts', 'Care of Magical Creatures', 'Arithmancy', 'year']
# UNNESSESARY_FEATURES = ['Best Hand', 'Arithmancy', 'Astronomy', 'Herbology',
#                         'Defense Against the Dark Arts', 'Divination',
#                         'Muggle Studies', 'Ancient Runes', 'History of Magic',
#                         'Transfiguration', 'Potions', 'Care of Magical Creatures',
#                         #'Charms', 'Flying',
#                         'year']


def binary():
    df_train = pd.read_csv('dataset_train.csv', index_col='Index')

    # craete year columns that shouws us cources of students
    df_train['year'] = df_train['Birthday'].apply(lambda x: int(x.split('-')[0]))
    df_train['year'] = df_train['year'] - df_train['year'].min()

    prepared_df = prepare_dataframe(df=df_train, drop_features=UNNESSESARY_FEATURES)

    binary_df = FrameHandler.cut_features(prepared_df, ['Hufflepuff',
                                                        'Ravenclaw',
                                                        'Slytherin',
                                                        # 'Gryffindor'
                                                        ])
    model = LogisticRegressionBinary(binary_df)
    model.set_target_column('Gryffindor')
    theta = LogisticRegressionBinary.fit(model.cost,
                                         model.theta.values.T,
                                         model.cost_gradient,
                                         model.X.values,
                                         model.y.values)

    res = LogisticRegressionBinary.predict(theta, model.X.values, model.y.values)
    res_list = res.flatten().tolist()
    right = len([i for i in res_list if i is True])
    print(f'True = ', right)
    wrong = len([i for i in res_list if i is False])
    print(f'False = ', wrong)
    print(wrong / len(res_list))
    print(right / len(res_list))


def multinomial():
    df_train = pd.read_csv('dataset_train.csv', index_col='Index')

    # craete year columns that shouws us cources of students
    df_train['year'] = df_train['Birthday'].apply(lambda x: int(x.split('-')[0]))
    df_train['year'] = df_train['year'] - df_train['year'].min()

    prepared_df = prepare_dataframe(df=df_train, drop_features=UNNESSESARY_FEATURES)

    model = LogisticRegressionMultinomial(prepared_df)
    model.set_target_column(['Hufflepuff', 'Ravenclaw', 'Slytherin', 'Gryffindor'])
    theta = LogisticRegressionMultinomial.fit(model.cost,
                                         model.theta.values.T,
                                         model.cost_gradient,
                                         model.X.values,
                                         model.y.values)

    res = LogisticRegressionMultinomial.predict(theta, model.X.values, model.y.values)
    res_list = res.flatten().tolist()
    right = len([i for i in res_list if i is True])
    print(f'True = ', right)
    wrong = len([i for i in res_list if i is False])
    print(f'False = ', wrong)
    print(wrong / len(res_list))
    print(right / len(res_list))


if __name__ == '__main__':
    # import data
    # binary()
    multinomial()