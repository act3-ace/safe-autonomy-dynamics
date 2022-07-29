from ruamel.yaml import YAML
import numpy as np

input_test_file = 'test_cases/CWHRotation2dSpacecraft_test_cases.yaml'
output_test_file = 'test_cases/CWHRotation2dSpacecraft_test_cases_y_thrust.yaml'


def angle_wrap(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

yaml=YAML(typ='rt')

with open(input_test_file, 'r') as input_f:
    input_tests = yaml.load(input_f)

output_tests = []
for test in input_tests:
    if 'control' not in test:
        continue
    
    control = test['control']

    if control[0] != 0 and control[1] == 0:
        test['ID'] = test['ID'] + '_y_thrust_only'
        test['control'][1] = control[0]
        test['control'][0] = 0

        cur_theta = test['attr_init']['theta']
        target_theta = test['attr_targets']['theta']

        if isinstance(cur_theta, list):
            cur_theta = cur_theta[0]
        if isinstance(target_theta, list):
            target_theta = target_theta[0]

        test['attr_init']['theta'] = angle_wrap(cur_theta - (np.pi/2))
        test['attr_targets']['theta'] = angle_wrap(target_theta - (np.pi/2))

        output_tests.append(test)

with open(output_test_file, 'w') as output_f:
    yaml.dump(output_tests, output_f)