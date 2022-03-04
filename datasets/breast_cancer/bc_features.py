from numpy import pi


label = {
         'no-recurrence-events': 0,
         'recurrence-events': 1,
         }

age = {
       '10-19': 0*pi/8,
       '20-29': 1*pi/8,
       '30-39': 2*pi/8,
       '40-49': 3*pi/8,
       '50-59': 4*pi/8,
       '60-69': 5*pi/8,
       '70-79': 6*pi/8,
       '80-89': 7*pi/8,
       '90-99': 8*pi/8,
       }

menopause = {
             'lt40': 0*pi/2,
             'ge40': 1*pi/2,
             'premeno': 2*pi/2,
             }

tumor_size = {
              '0-4': 0*pi/11,
              '5-9': 1*pi/11,
              '10-14': 2*pi/11,
              '15-19': 3*pi/11,
              '20-24': 4*pi/11,
              '25-29': 5*pi/11,
              '30-34': 6*pi/11,
              '35-39': 7*pi/11,
              '40-44': 8*pi/11,
              '45-49': 9*pi/11,
              '50-54': 10*pi/11,
              '55-59': 11*pi/11,
              }

inv_nodes = {
             '0-2': 0*pi/12,
             '3-5': 1*pi/12,
             '6-8': 2*pi/12,
             '9-11': 3*pi/12,
             '12-14': 4*pi/12,
             '15-17': 5*pi/12,
             '18-20': 6*pi/12,
             '21-23': 7*pi/12,
             '24-26': 8*pi/12,
             '27-29': 9*pi/12,
             '30-32': 10*pi/12,
             '33-35': 11*pi/12,
             '36-39': 12*pi/12,
             }

node_caps = {
             'yes': 0*pi/1,
             'no': 1*pi/1,
             }

deg_malig = {
             '1': 0*pi/2,
             '2': 1*pi/2,
             '3': 2*pi/2,
             }

breast = {
          'left': 0*pi/1,
          'right': 1*pi/1,
          }

breast_quad = {
               'left_up': 0*pi/4,
               'left_low': 1*pi/4,
               'right_up': 2*pi/4,
               'right_low': 3*pi/4,
               'central': 4*pi/4,
               }

irradiat = {
            'yes\n': 0*pi/1,
            'no\n': 1*pi/1,
            }

bc_features = [label, age, menopause, tumor_size, inv_nodes,
               node_caps, deg_malig, breast, breast_quad, irradiat]
