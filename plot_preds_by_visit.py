import pickle
import matplotlib.pyplot as plt

preds = pickle.load(open('results/temporal_completion_stats/MediSim_Metrics.pkl', 'rb'))
values = preds['F1 Score'][:-1]
plt.plot(list(range(1, 99)), values)
plt.xlabel('Visit Number', color='black')
plt.ylabel('F1 Score', color='black')
plt.title('Average Test F1 Score by Visit Number', color='black')
plt.grid(b=None)
ax = plt.gca()
ax.set_facecolor('white')
plt.savefig('results/temporal_completion_stats/accuracyOverTime.png')