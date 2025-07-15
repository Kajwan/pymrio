"""Test cases for HEM calculations."""


import os
import sys

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest

TESTPATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(TESTPATH, ".."))


# the function which should be tested here
from pymrio.tools.iohem import HEM

def td_small_MRIO():
    """Small MRIO with three sectors and two regions.

    The testdata here just consists of pandas DataFrames, the functionality
    with numpy arrays gets tested with td_IO_Data_Miller.
    """

    class IO_Data:
        _sectors = ["sector1", "sector2", "sector3"]
        _regions = ["reg1", "reg2"]
        _Z_multiindex = pd.MultiIndex.from_product([_regions, _sectors], names=["region", "sector"])

        Z = pd.DataFrame(
            data=[
                [10, 5, 1, 6, 5, 7],
                [0, 2, 0, 0, 5, 3],
                [10, 3, 20, 4, 2, 0],
                [5, 0, 0, 1, 10, 9],
                [0, 10, 1, 0, 20, 1],
                [5, 0, 0, 1, 10, 10],
            ],
            index=_Z_multiindex,
            columns=_Z_multiindex,
            dtype=("float64"),
        )

        _categories = ["final demand"]
        _Y_multiindex = pd.MultiIndex.from_product([_regions, _categories], names=["region", "category"])
        Y = pd.DataFrame(
            data=[[14, 3], [2.5, 2.5], [13, 6], [5, 20], [10, 10], [3, 10]],
            index=_Z_multiindex,
            columns=_Y_multiindex,
            dtype=("float64"),
        )

        F = pd.DataFrame(
            data=[[20, 1, 42, 4, 20, 5], [5, 4, 11, 8, 2, 10]],
            index=["ext_type_1", "ext_type_2"],
            columns=_Z_multiindex,
            dtype=("float64"),
        )

        F_Y = pd.DataFrame(
            data=[[50, 10], [100, 20]],
            index=["ext_type_1", "ext_type_2"],
            columns=_Y_multiindex,
            dtype=("float64"),
        )

        S_Y = pd.DataFrame(
            data=[
                [1.0526315789473684, 0.1941747572815534],
                [2.1052631578947367, 0.3883495145631068],
            ],
            index=["ext_type_1", "ext_type_2"],
            columns=_Y_multiindex,
            dtype=("float64"),
        )

        A = pd.DataFrame(
            data=[
                [
                    0.19607843137254902,
                    0.3333333333333333,
                    0.017241379310344827,
                    0.12,
                    0.09615384615384616,
                    0.1794871794871795,
                ],  # noqa
                [
                    0.0,
                    0.13333333333333333,
                    0.0,
                    0.0,
                    0.09615384615384616,
                    0.07692307692307693,
                ],  # noqa
                [
                    0.19607843137254902,
                    0.2,
                    0.3448275862068966,
                    0.08,
                    0.038461538461538464,
                    0.0,
                ],  # noqa
                [
                    0.09803921568627451,
                    0.0,
                    0.0,
                    0.02,
                    0.19230769230769232,
                    0.23076923076923075,
                ],  # noqa
                [
                    0.0,
                    0.6666666666666666,
                    0.017241379310344827,
                    0.0,
                    0.38461538461538464,
                    0.02564102564102564,
                ],  # noqa
                [
                    0.09803921568627451,
                    0.0,
                    0.0,
                    0.02,
                    0.19230769230769232,
                    0.2564102564102564,
                ],  # noqa
            ],
            index=_Z_multiindex,
            columns=_Z_multiindex,
        )

        L = pd.DataFrame(
            data=[
                [
                    1.3387146304736708,
                    0.9689762471208287,
                    0.05036622549592462,
                    0.17820960407435948,
                    0.5752019383714646,
                    0.4985179148178926,
                ],  # noqa
                [
                    0.02200779585580331,
                    1.3716472861392823,
                    0.0076800357678581885,
                    0.006557415453762468,
                    0.2698335633228079,
                    0.15854643902810828,
                ],  # noqa
                [
                    0.43290422861412026,
                    0.8627066565439678,
                    1.5492942759220427,
                    0.18491657196329184,
                    0.44027825642348534,
                    0.26630955082840885,
                ],  # noqa
                [
                    0.18799498787612925,
                    0.5244084722329316,
                    0.020254008037620782,
                    1.0542007368783255,
                    0.5816573175534603,
                    0.44685014763069275,
                ],  # noqa
                [
                    0.04400982046095892,
                    1.5325472495862535,
                    0.05259311578831879,
                    0.014602513642445088,
                    1.9545285794951548,
                    0.2410917825607805,
                ],  # noqa
                [
                    0.19294222439918532,
                    0.5382086951864299,
                    0.020787008249137116,
                    0.05562707205933412,
                    0.596964089068025,
                    1.4849251515157111,
                ],  # noqa
            ],
            index=_Z_multiindex,
            columns=_Z_multiindex,
        )


        x = pd.DataFrame(
            data=[
                [51],
                [15],
                [58],
                [50],
                [52],
                [39],
            ],
            columns=["indout"],
            index=_Z_multiindex,
            dtype=("float64"),
        )
        S = pd.DataFrame(
            data=[
                [
                    0.39215686274509803,
                    0.06666666666666667,
                    0.7241379310344828,
                    0.08,
                    0.38461538461538464,
                    0.1282051282051282,
                ],  # noqa
                [
                    0.09803921568627451,
                    0.26666666666666666,
                    0.1896551724137931,
                    0.16,
                    0.038461538461538464,
                    0.2564102564102564,
                ],  # noqa
            ],
            index=["ext_type_1", "ext_type_2"],
            columns=_Z_multiindex,
        )

    return IO_Data


def test_hem_extraction(td_small_mrio, regions=["reg1"], sectors=["sector1", "sector2"]):
    """Test the extraction of HEM data from a small MRIO."""
    IO_Data = td_small_MRIO.A
    HEM_object = HEM(IOSystem=None, Y=td_small_MRIO.Y, A=td_small_MRIO.A, x=td_small_MRIO.x, L=td_small_MRIO.L, meta=None, save_path=None)
    HEM_object.make_extraction(regions=["reg1"], sectors=["sector1", "sector2"], extraction_type="1.2", multipliers=True)
    pdt.assert_frame_equal(
        left=IO_Data.x.loc[HEM_object.index_extraction, "indout"],
        right=HEM_object.production.sum(axis=1)
    )

def test_hem_extraction_impacts(td_small_mrio, regions=["reg1"], sectors=["sector1", "sector2"]):
    """Test the extraction of HEM data from a small MRIO."""
    IO_Data = td_small_MRIO.A
    HEM_object = HEM(IOSystem=None, Y=td_small_MRIO.Y, A=td_small_MRIO.A, x=td_small_MRIO.x, L=td_small_MRIO.L, meta=None, save_path=None)
    HEM_object.make_extraction(regions=["reg1"], sectors=["sector1", "sector2"], extraction_type="1.2", multipliers=True)
    HEM_object.calculate_impacts(IO_Data.S)

    pdt.assert_frame_equal(
        left=IO_Data.F.loc[:,HEM_object.index_extraction].sum(axis=1),
        right=HEM_object.impact_production.sum(axis=1)
    )
