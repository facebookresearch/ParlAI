#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd


RACES_ETHNICITIES = ['hispanic', 'white', 'black', 'api', 'aian', '2prace']
# Notations for races/ethnicities reflect those used in Tzioumis et al. (see
# https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/TYJKEZ/
# MPMHFE&version=1.3 for details)
RACES_ETHNICITIES_WITH_NAMES = ['hispanic', 'white', 'black', 'api']
# Some races/ethnicities don't have any names on the Tzioumis list for which they are
# the plurality race/ethnicity; we exclude those races/ethnicities from this list.


def get_gender_name_list(gender: str, names_path: str) -> List[str]:
    """
    Return a list of names of the specified gender from Newman, et al.

    Read names from https://journals.sagepub.com/doi/abs/10.1177/0146167218769858 and
    filter by the specified gender.
    """
    name_df = pd.read_csv(names_path)
    names = name_df[lambda df: df['Gender'].str.lower() == gender][
        'Name'
    ].values.tolist()
    print(f'Using {len(names):d} {gender} names: ' + ', '.join(names))
    return names


def get_race_ethnicity_gender_name_list(
    baby_name_folder: str,
    tzioumis_data_path: str,
    race_gender_name_lists: Dict[str, List[str]],
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Get name lists split by both race/ethnicity and by gender, given input name lists
    (see inner functions for the sources of these lists).
    """

    baby_name_counts_by_gender = get_baby_name_counts_by_gender(baby_name_folder)

    # Split original name lists by gender
    orig_name_lists = get_tzioumis_name_lists(
        tzioumis_data_path=tzioumis_data_path,
        race_gender_name_lists=race_gender_name_lists,
    )
    print('\nSplitting original name lists by gender.')
    names_to_new_lists = {}
    for name_list, names in orig_name_lists.items():
        for name in names:
            proc_name = name.replace('-', '')
            proc_name = proc_name[0].upper() + proc_name[1:].lower()
            # Removing hyphens and changing capitalization to match the formatting of
            # the baby-name lists
            if (
                baby_name_counts_by_gender[proc_name]['F']
                > baby_name_counts_by_gender[proc_name]['M']
            ):
                names_to_new_lists[name] = f'{name_list}_female'
            elif (
                baby_name_counts_by_gender[proc_name]['M']
                > baby_name_counts_by_gender[proc_name]['F']
            ):
                names_to_new_lists[name] = f'{name_list}_male'
            else:
                # Tie
                names_to_new_lists[name] = f'{name_list}_unknown'
    assert len(names_to_new_lists) == sum(
        [len(names) for names in orig_name_lists.values()]
    )

    # Print the names in each of the new name lists
    new_name_lists = defaultdict(list)
    for name, name_list in names_to_new_lists.items():
        new_name_lists[name_list].append(name)
    for name_list in sorted(new_name_lists.keys()):
        sorted_names = sorted(new_name_lists[name_list])
        print(
            f'\nUsing {len(sorted_names):d} names for the {name_list} name list: '
            + ', '.join(sorted_names)
        )

    return names_to_new_lists, new_name_lists


def get_baby_name_counts_by_gender(baby_name_folder: str) -> Dict[str, Dict[str, int]]:
    """
    Return a dictionary whose keys are baby names and whose values are counts of the
    number of babies given that name, split by gender.

    Baby name folder from https://catalog.data.gov/dataset/
    baby-names-from-social-security-card-applications-national-data,
    accessed 2021-04-02.
    """

    # Params
    final_baby_name_year = 2019
    baby_name_year_range = range(final_baby_name_year - 99, final_baby_name_year + 1)
    # Get the most recent 100 years of names

    # Get counts of baby names by gender
    baby_name_counts_by_gender = defaultdict(lambda: {'F': 0, 'M': 0})
    for year in baby_name_year_range:
        counts_path = os.path.join(baby_name_folder, f'yob{year:d}.txt')
        with open(counts_path) as f:
            for line in f:
                name, gender, count_string = line.split(',')
                count = int(count_string.rstrip())
                baby_name_counts_by_gender[name][gender] += count

    return baby_name_counts_by_gender


def get_tzioumis_name_lists(
    tzioumis_data_path: str,
    race_gender_name_lists: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """
    Get race/ethnicity name lists from the Tzioumis work.
    """
    percent_df = load_tzioumis_data(tzioumis_data_path)
    name_lists = {}
    for race_ethnicity in RACES_ETHNICITIES_WITH_NAMES:
        name_lists[race_ethnicity] = get_race_ethnicity_name_list_given_tzioumis_data(
            percent_df=percent_df,
            race_gender_name_lists=race_gender_name_lists,
            race_ethnicity=race_ethnicity,
        )
    return name_lists


def load_tzioumis_data(tzioumis_data_path: str) -> pd.DataFrame:
    """
    Load Tzioumis data from
    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FTYJKEZ
    (Paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5839157/)
    """
    print(f'Loading Tzioumis data from {tzioumis_data_path}')
    percent_df = (
        pd.read_excel(tzioumis_data_path, sheet_name='Data')[
            lambda df: df['firstname'] != 'ALL OTHER FIRST NAMES'
        ]
        .assign(
            firstname=lambda df: df['firstname'].apply(lambda s: s[0] + s[1:].lower())
        )
        .set_index('firstname')
    )
    return percent_df


def get_race_ethnicity_name_list_given_tzioumis_data(
    percent_df: pd.DataFrame,
    race_gender_name_lists: Dict[str, List[str]],
    race_ethnicity: str,
) -> List[str]:
    """
    Given input Tzioumis data (percent_df) and a set of names split by gender and
    race/ethnicity from Milkman et al.

    (2012), Caliskan et al. (2017), and Guo and Caliskan (2020)., get a list of names
    for the given race/ethnicity.
    """

    # Params
    tzioumis_to_race_gender_mapping = {'hispanic': 'his', 'white': 'ea', 'black': 'aa'}
    num_names_per_race_ethnicity = 200
    # Number of names to select per race/ethnicity, to keep the lists tractable

    # Determine which names are most commonly of the specified race/ethnicity (i.e.
    # plurality), and pick those that have the most observations for that
    # ethnicity
    this_ethnicity_column = f'pct{race_ethnicity}'
    percent_columns = [f'pct{race_eth}' for race_eth in RACES_ETHNICITIES]
    max_percent_series = percent_df[percent_columns].max(axis=1)
    percent_plurality_names_df = (
        percent_df[lambda df: df[this_ethnicity_column] == max_percent_series]
        .assign(
            obs_of_this_ethnicity=lambda df: df['obs'] * df[this_ethnicity_column] / 100
        )
        .sort_values('obs_of_this_ethnicity', ascending=False)
    )
    tzioumis_plurality_names = percent_plurality_names_df.iloc[
        :num_names_per_race_ethnicity
    ].index.values.tolist()

    # Combine these names with the Caliskan+ race+gender names and deduplicate
    if race_ethnicity in tzioumis_to_race_gender_mapping:
        mapped_ethnicity = tzioumis_to_race_gender_mapping[race_ethnicity]
        female_race_gender_name_list = race_gender_name_lists[
            f'{mapped_ethnicity}_female'
        ]
        if mapped_ethnicity == 'aa':
            # Avoid the same name in two lists by removing it from this one
            female_race_gender_name_list.remove('Yolanda')
        elif mapped_ethnicity == 'his':
            # Avoid the same name in two lists by removing it from this one
            female_race_gender_name_list.remove('Brenda')
        # TODO: add a programmatic way to detect duplicates, to generalize for updated
        #  versions of the source datasets
        male_race_gender_name_list = race_gender_name_lists[f'{mapped_ethnicity}_male']
        combined_names = (
            tzioumis_plurality_names
            + female_race_gender_name_list
            + male_race_gender_name_list
        )
    else:
        combined_names = tzioumis_plurality_names

    # Deduplicate and sort
    sorted_names = sorted(list(set(combined_names)))

    print(
        f'Using {len(sorted_names):d} names for the {race_ethnicity} race/ethnicity: '
        + ', '.join(sorted_names)
    )

    return sorted_names
