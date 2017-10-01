import pytest
import matchteller as mt

def test_pre_calculated_poisson_prediction():
    """ """
    p = mt.PoissonPredictor(['../2015-2016/E0.csv'])

    p.calc()

    outcome = p.predict('Stoke', 'Arsenal')

    assert outcome['AWAY'], 17.588533
