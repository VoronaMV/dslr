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

    res = LogisticRegressionMultinomial.predict(theta.T, model.X.values, model.y.values)

    theta_df = pd.DataFrame(theta, columns=model.theta.columns.tolist())
    theta_df.to_csv('res.csv', index=False)

    res_list = res.flatten().tolist()
    right = len([i for i in res_list if i is True])
    print(f'True = ', right)
    wrong = len([i for i in res_list if i is False])
    print(f'False = ', wrong)
    print(wrong / len(res_list))
    print(right / len(res_list))


def predict():
    df_test = pd.read_csv('dataset_test.csv', index_col='Index')
    df_test['year'] = df_test['Birthday'].apply(lambda x: int(x.split('-')[0]))
    df_test['year'] = df_test['year'] - df_test['year'].min()
    prepared_df = prepare_dataframe(df=df_test, drop_features=UNNESSESARY_FEATURES)
    model = LogisticRegressionMultinomial(prepared_df)
    # model.set_target_column(['Hufflepuff', 'Ravenclaw', 'Slytherin', 'Gryffindor'], create=True)
    model.X = model.frame.copy()

    predicted_theta = pd.read_csv('res.csv')
    predictions = LogisticRegressionMultinomial.real_predict(predicted_theta.values.T, model.X.values)
    print()


if __name__ == '__main__':
    # import data
    # binary()
    multinomial()
    predict()
