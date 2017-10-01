import pandas as pd
import numpy as np
import scipy.stats as st

from .team import Teams


class ModelBase(object):

    def given(self, data=[]):
        try:
            # Load and store the dataset(s)
            list_ = [pd.read_csv(path) for path in data]

            # Merge all data sources into a single dataframe
            self.d = pd.concat(list_)

            # format the 'Date' column
            self.d['Date'] = pd.to_datetime(self.d['Date'], dayfirst=True)

            # Order the date by the newly formated date
            self.d = self.d.sort_values(by='Date', ascending=False).reset_index()

            # Rename the index column (original position in CSV), created by
            # reset_index() above, to (S)eason (G)ame (N)umber
            self.d = self.d.rename(index=str, columns={"index": "SGN"})
        except ValueError:
            raise ValueError('You did not provide a list of CSV datasets')
        except AttributeError:
            raise AttributeError('The CSV dataset is malformed')

        return self

    def when(self, name):
        """ Set the home team """
        self.teams = Teams(home=name, data=self.d)
        return self

    def played(self, name):
        """ Set the away team """
        self.teams = Teams(away=name, data=self.d)
        return self

    def last(self):
        pass

    def on(self, date):
        """ Set the date on which the match prediction will occur """
        self.m_dates = pd.Series([date], index=['Date'], dtype='datetime64[ns]')
        return self


class Poisson(ModelBase):
    """ Poisson Model to Predict the out of Associate Football Matches"""

    def __init__(self):
        """ Initialise the Predictor"""
        self.m_dates = None

    def predict(self):
        """ Predict probability of the matches final outcome"""

        if self.m_dates is None:
            self._calc_match_dates()

        self._data_selection_strat()

        self._calc_home_team_advant()

        self._calc_score_mtx()

        self._calc_outcome_probs()
        self._calc_outcome_odds()

        return self.m_outcome_prob

    def result(self):
        """ Return the actual result of the match """
        return self.d[
            (self.d['HomeTeam'] == self.teams.home) &
            (self.d['AwayTeam'] == self.teams.away)
        ][['FTHG', 'FTAG', 'FTR']]

    def _data_selection_strat(self, offset=0, limit=380):
        """ """
        self._d = self.d[
            self.d.index.astype(int) < int(self.m_dates.index[offset])
        ].head(limit)

    def _calc_match_dates(self):
        """ Get the date of the last match of home_team vs away_team """
        # FIXME: This method should return a single date, , or the most recent by default
        #        if a match date was provided, or an offset from the most recent
        if self.m_dates is not None:
            self.m_dates = self.d[
                (self.d['HomeTeam'] == self.teams.home) &
                (self.d['AwayTeam'] == self.teams.away) &
                (self.d['Date'] == self.match_dates.loc[1]['Date'])
            ]['Date']
        else:
            self.m_dates = self.d[
                (self.d['HomeTeam'] == self.teams.home) &
                (self.d['AwayTeam'] == self.teams.away)
            ]['Date']

    def _calc_home_team_advant(self):
        """ Calculate the home team advantage """
        self.home_team_advant = (
            self._d.groupby('FTR')['FTR'].count()['H'] /
            ((self._d.groupby('FTR')['FTR'].count()['H'] + self._d.groupby('FTR')['FTR'].count()['A']) / 2)
        )

    def _calc_score_mtx(self):
        """ Calculate the probability of each score outcome """
        self.m_score = pd.DataFrame(
            np.zeros((11, 11), dtype=int),
            index=np.arange(11)
        )
        self.teams.stats()
        self.m_score = self.m_score.apply(
            lambda x: x + (
                (st.poisson.pmf(x.index + 1, self.home_team_advant * self.teams.goals_exp.loc[self.teams.home]['HGE'], 1)) *
                (st.poisson.pmf(x.name + 1, self.teams.goals_exp.loc[self.teams.away]['AGE'] / self.home_team_advant, 1)) *
                100
            )
        )

    def _calc_outcome_probs(self):
        """ Calculate the probability """
        self.m_outcome_prob = pd.DataFrame({
            'HOME': self.m_score[self.m_score.apply(lambda x: x.name < x.index)].sum(axis=0).sum(),
            'DRAW': self.m_score[self.m_score.apply(lambda x: x.name == x.index)].sum().sum(),
            'AWAY': self.m_score[self.m_score.apply(lambda x: x.name > x.index)].sum().sum()
        }, index=['PROB'])

    def _calc_outcome_odds(self):
        """ Calculate the odds """
        self.m_outcome_odds = pd.DataFrame({
            'HOME': (1/self.m_outcome_prob['HOME']) * 100,
            'DRAW': (1/self.m_outcome_prob['DRAW']) * 100,
            'AWAY': (1/self.m_outcome_prob['AWAY']) * 100
        }, index=['ODDS'])
