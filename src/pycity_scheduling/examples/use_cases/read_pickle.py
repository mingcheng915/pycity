import cloudpickle
import matplotlib.pyplot as plt
from pycity_scheduling.util.district_analyzer import DistrictAnalyzer

file = open('U:\\ACS-Public\\Personal_Folders\\ssw\\cvo\\results_cluster\\20221108_infeasibility_test\\store_analyzer_miqp', 'rb')

d_a = cloudpickle.load(file)
d_a.count_violated_constraints()
d_a.plot_city_district_schedule()
d_a.complete_post_analyze()
plt.show()
