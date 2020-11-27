import json
import re
import pickle
import time
from datetime import datetime

import requests
import pandas as pd


def get_star_time(df, j):
    star_path = 'https://api.github.com/repos/{}/{}/stargazers'
    token = 'a778aeacfadcaae486f97b8c9a560121b4b44711'
    Accept = 'application/vnd.github.v3.star+json'
    for package_name, aha in df.github.groupby(level=0):
        for i in aha.values[::-1]:
            if i not in (None, True, False):
                star_res = requests.get(star_path.format(*i), headers={'Accept': Accept, 'Authorization': 'token ' + token})
                msg = json.loads(star_res.text)
                try:
                    j.append(dict(
                        _id=package_name,
                        star_time=[i['starred_at'] for i in msg]
                    ))
                except TypeError:
                    if msg['message'] == 'Not Found':
                        continue
                    elif 'rate limit' in msg['message']:
                        raise ValueError(star_res.headers['X-RateLimit-Reset'])
                    else:
                        print(msg)
                        return None
                else:
                    break
    return None


if __name__ == '__main__':
    df = pd.read_pickle('pypi_json.pkl')
    packages = sorted(set(list(zip(*df.index))[0]))
    # j1, j2, j3 = [], [], []
    j = []
    start, goal = 1293, 10000
    
    while True:
        my_df = df.loc[packages[start:goal]]
        try:
            get_star_time(my_df, j)
        except ValueError as reset:
            now = time.time()
            wait = int(reset) - now
            print(f'request rate limit reached at {datetime.now():%l:%M%p on %b %d, %Y}, '
                  f'and will wait for {wait} seconds until {datetime.fromtimestamp(reset):%l:%M%p on %b %d, %Y}')
            start = packages.index(j[-1]['_id'])
            time.sleep(wait)
            continue
        except ConnectionError:
            start = packages.index(j[-1]['_id'])
        else:
            break