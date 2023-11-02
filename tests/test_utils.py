from data_selection import utils


def job(arg):
    return arg + 1


def test_parallelize():

    results = utils.parallelize(job, [1, 2, 3, 4, 5], num_proc=2)
    assert sum(results) == sum([2, 3, 4, 5, 6])
