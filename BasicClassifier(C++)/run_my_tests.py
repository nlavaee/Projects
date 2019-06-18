#! /usr/bin/env python

import os
import sys
import subprocess

from fnmatch import fnmatch

print('Running {0} tests...'.format(sys.argv[1]))

test_exe_pattern = sys.argv[1] + '_test*.exe'

test_exes = [filename for filename in os.listdir('.')
             if fnmatch(filename, test_exe_pattern)]

num_tests_run = 0
num_tests_passed = 0

valgrind_results = 'Unable to run valgrind: valgrind not installed'

for test in test_exes:
    print('*** Starting test {0} ***'.format(test))
    return_code = subprocess.call(['./' + test])
    num_tests_run += 1
    if return_code == 0:
        num_tests_passed += 1
    print('*** Test {0} {1} ***'.format(test,
                                        'failed' if return_code else 'passed'))

if subprocess.call(['which', 'valgrind']) == 0:
    num_valgrind_errors = 0
    for test in test_exes:
        valgrind_return_code = subprocess.call(
            ['valgrind', '--leak-check=full', '--error-exitcode=1',
             './' + test])
        if valgrind_return_code != 0:
            num_valgrind_errors += 1
            print('valgrind error in test: ' + test)
    valgrind_results = '{0} valgrind errors'.format(num_valgrind_errors)

print('''Out of {0} tests run:
{1} tests passed,
{2} tests failed.
{3}
'''.format(
    num_tests_run,
    num_tests_passed,
    num_tests_run - num_tests_passed,
    valgrind_results))
