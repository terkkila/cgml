

from nose.tools import assert_equals

from cgml.parsers.base import _make_camelcase

def test_make_camelcase():
    
    assert_equals( _make_camelcase("absolute-percentage-error"), "absolutePercentageError" )
