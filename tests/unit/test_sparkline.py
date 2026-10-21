from vigilex_api.pages import sparkline


def test_sparkline_scales_to_box():
    assert sparkline([0.0, 0.5, 1.0], width=100, height=10) == "0.0,10.0 50.0,5.0 100.0,0.0"


def test_flat_and_short_series():
    assert sparkline([0.3, 0.3], width=10, height=10) == "0.0,5.0 10.0,5.0"
    assert sparkline([0.3]) == "" and sparkline([]) == ""
