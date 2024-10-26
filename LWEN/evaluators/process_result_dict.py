from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from sklearn.metrics import r2_score

plt.rc('font', family='FreeSerif')
_exist_date = ['03/08', '03/11', '03/15', '03/22', '03/29', '04/01', '04/04', '04/08', '04/12', '04/15', '04/19',
               '04/22', '04/26', '04/29', '05/03', '05/06', '05/13', '05/17', '05/20', '05/24', '05/29']


def initial_dict(id_list):
    data = defaultdict(lambda: {})
    for i in id_list:
        if i == 40:
            continue
        for date in _exist_date:
            data[str(i)][date] = defaultdict(list)
    return data


def mean(lst):
    return sum(lst) / len(lst)


def caculate_mean(dic):
    global_predict = defaultdict(list)
    global_true = defaultdict(list)

    for pig_id, date_dic in dic.items():
        for date, data in date_dic.items():
            if 'predict' in data.keys():
                global_predict[date].extend(data['predict'][:])
                global_true[date].append(data['true'])
                dic[pig_id][date]['pic_number'] = len(data['predict'])
                dic[pig_id][date]['predict'] = mean(data['predict'])

    over_mean = defaultdict(lambda: {})
    for date in global_predict:
        over_mean[date]['pic_number'] = len(global_predict[date])
        over_mean[date]['predict'] = mean(global_predict[date])
        over_mean[date]['true'] = mean(global_true[date])
    return over_mean


def draw_stat_fig(dic, title):
    x = [datetime.strptime(i, '%m/%d').date() for i in dic]
    y_predict = [i['predict'] for _, i in dic.items()]
    y_true = [i['true'] for _, i in dic.items()]
    y_number = [i['pic_number'] for _, i in dic.items()]

    fig, ax = plt.subplots(figsize=[10, 6], dpi=200)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.set_xticks(x)
    fig.autofmt_xdate()
    ax.plot(x, y_predict, color='#2878b5', marker='.', label='predict')
    ax.plot(x, y_true, color='#c82423', marker='.', label='true')
    ax.legend(loc='center right', fontsize=14)
    # ax.tick_params(labelsize=10)
    # 为每个点算出abs—bias
    error = []
    for xaxis, y_p, y_t in zip(x, y_predict, y_true):
        height = max(y_p, y_t)
        bias = abs(y_p - y_t)
        error.append(bias)
        ax.text(xaxis, height, '{:.2f}'.format(bias), ha='center', va='top')

    # 绘制每个日期图片数
    ax_r = ax.twinx()
    ax_r.bar(x, y_number, color='#ff8884', alpha=0.3, label='pic_number')
    ax_r.legend(loc='lower right', fontsize=14)
    for xaxis, h in zip(x, y_number):
        ax_r.text(xaxis, h, h, alpha=0.3, ha='center')

    # 设置坐标轴标签
    ax.set_xlabel('Date', fontsize=14, loc='center')
    ax.set_ylabel('PigWeight', fontsize=14, loc='center')
    ax_r.set_ylabel('Pic_number', fontsize=14, loc='center')
    # 计算MAE，R2
    mae = mean(error)
    r2 = r2_score(y_predict, y_true)
    ax.set_title('{}, MAE: {:.2f}, R^2: {:.2f}'.format(title, mae, r2), loc='left', fontsize=16)
    fig.tight_layout()
    return fig
