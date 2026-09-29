from acs import config
from acs.pricing import total_input, total_output, usd_empirical, usd_nocache


def test_usd_nocache_accept():
    call = {"input_tokens": 10_000, "output_tokens": 1_000}
    assert usd_nocache(call) == 0.015


def test_usd_empirical_accept():
    call = {"input_tokens": 2_000, "cache_read_input_tokens": 8_000, "output_tokens": 1_000}
    assert abs(usd_empirical(call) - 0.0078) < 1e-12


def test_total_input_sums_the_three_input_fields():
    call = {
        "input_tokens": 1,
        "cache_creation_input_tokens": 2,
        "cache_read_input_tokens": 4,
        "output_tokens": 8,
    }
    assert total_input(call) == 7
    assert total_output(call) == 8


def test_nocache_is_never_cheaper_than_empirical():
    call = {"input_tokens": 100, "cache_creation_input_tokens": 500,
            "cache_read_input_tokens": 9_000, "output_tokens": 50}
    assert usd_nocache(call) > usd_empirical(call)


def test_pricing_constants_are_the_preregistered_ones():
    assert config.PRICE_INPUT_PER_MTOK == 1.00
    assert config.PRICE_OUTPUT_PER_MTOK == 5.00
    assert config.PRICE_CACHE_WRITE_PER_MTOK == 1.25
    assert config.PRICE_CACHE_READ_PER_MTOK == 0.10
    assert config.CACHE_MIN_TOKENS["claude-haiku-4-5"] == 4096
