import pytest

from appworld.common.errors import MaybeSuppressErrors, NoSuppressError


class TestCommonErrors:
    def test_error_suppressor(self) -> None:
        with MaybeSuppressErrors(suppress_errors=True) as suppressor:
            raise Exception("An error")
        assert suppressor.raised_error
        assert str(suppressor.exception) == "An error"
        with pytest.raises(Exception):  # noqa: B017
            with MaybeSuppressErrors(suppress_errors=False) as suppressor:
                raise Exception("An error")
        for suppress_errors in [True, False]:
            with MaybeSuppressErrors(suppress_errors=suppress_errors) as suppressor:
                pass
            assert not suppressor.raised_error
            assert not suppressor.exception
        with pytest.raises(Exception):  # noqa: B017
            with MaybeSuppressErrors(suppress_errors=False) as suppressor:
                raise NoSuppressError()
        with pytest.raises(Exception):  # noqa: B017
            with MaybeSuppressErrors(suppress_errors=False) as suppressor:
                raise NoSuppressError()
        for suppress_errors in [True, False]:
            with pytest.raises(NoSuppressError) as exception:
                with MaybeSuppressErrors(suppress_errors=suppress_errors) as _:
                    raise NoSuppressError("This is no suppression error.")
            assert str(exception.value) == "This is no suppression error."
