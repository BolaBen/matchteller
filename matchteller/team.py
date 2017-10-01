import pandas as pd
import scipy.stats as st


class Borg:
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state


class Teams(Borg):
    """ """
    def __init__(self, home=None, away=None, data=None):
        Borg.__init__(self)

        if home is not None:
            self.home = home
        if away is not None:
            self.away = away

        self._d = data

    def stats(self):
        """ Calculate the teams statistics """
        self._scores()
        self._strengths()
        self._goal_exp()

        return {
            'teams': self.t,
            'home': self.home,
            'away': self.away,
            'scores': self.scores,
            'strengths': self.strengths,
            'expectation': self.goals_exp
        }

    def _scores(self):
        # Create a list of all teams in the dataset
        self.t = self._d.groupby('HomeTeam').groups.keys()
        self.scores = pd.DataFrame({
                # Total Goals
                'TG': pd.Series(
                    self._d.groupby('HomeTeam')['FTHG'].sum().values +
                    self._d.groupby('AwayTeam')['FTAG'].sum().values,
                    index=self.t
                ),
                # Total Home Goals
                'THG': pd.Series(
                    self._d.groupby('HomeTeam')['FTHG'].sum().values,
                    index=self.t
                ),
                # Average Home Goals
                'AHG': pd.Series(
                    self._d.groupby('HomeTeam')['FTHG'].mean().values,
                    index=self.t
                ),
                # Total Home Goals Conceded
                'THGC': pd.Series(
                    self._d.groupby('HomeTeam')['FTAG'].sum().values,
                    index=self.t
                ),
                # Average Home Goal Conceded
                'AHGC': pd.Series(
                    self._d.groupby('HomeTeam')['FTAG'].mean().values,
                    index=self.t
                ),
                # Total Away Goals
                'TAG': pd.Series(
                    self._d.groupby('AwayTeam')['FTAG'].sum().values,
                    index=self.t
                ),
                # Average Away Goals
                'AAG': pd.Series(
                    self._d.groupby('AwayTeam')['FTAG'].mean().values,
                    index=self.t
                ),
                # Total Away Goals Conceded
                'TAGC': pd.Series(
                    self._d.groupby('AwayTeam')['FTHG'].sum().values,
                    index=self.t
                ),
                # Average Away Goals Conceded
                'AAGC': pd.Series(
                    self._d.groupby('AwayTeam')['FTHG'].mean().values,
                    index=self.t
                ),
            }
        )

        self.scores_ttl = self.scores[['TG', 'THG', 'THGC', 'TAG', 'TAGC']].sum()
        self.scores_avg = self.scores[['AHG', 'AHGC', 'AAG', 'AAGC']].apply(axis=0, func=st.hmean)

    def _strengths(self):
        self.strengths = pd.DataFrame({
            'HAS': self.scores['AHG'] / self.scores_avg['AHG'],
            'HDS': self.scores['AHGC'] / self.scores_avg['AHGC'],
            'AAS': self.scores['AAG'] / self.scores_avg['AAG'],
            'ADS': self.scores['AAGC'] / self.scores_avg['AAGC']
        })

    def _goal_exp(self):
        self.goals_exp = pd.DataFrame({
            'HGE': self.scores['AHG'] * self.strengths['HAS'] * self.strengths['ADS'],
            'AGE': self.scores['AAG'] * self.strengths['HDS'] * self.strengths['AAS']
        })
