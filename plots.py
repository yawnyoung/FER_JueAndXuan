import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

expr_classes = ('anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise')
all_num = [4, 1, 5, 2, 6, 2, 8]
true_num_15 = [2, 1, 4, 1, 5, 1, 7]
true_num_1 = [4, 0, 5, 0, 5, 2, 7]

fig = plt.figure()

x_pos = list(range(len(expr_classes)))
ax1 = fig.add_subplot(121)
ax1.bar(x_pos, all_num, color='y', align='center')
ax1.bar(x_pos, true_num_1, color='g', align='center')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(expr_classes)
ax1.set_ylabel('Number of Images')
ax1.set_title('Small Loss Weights Adjustment')
ax1.grid(linestyle='--', alpha=0.5)

ax2 = fig.add_subplot(122)

ax2.bar(x_pos, all_num, color='y', align='center')
ax2.bar(x_pos, true_num_15, color='g', align='center')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(expr_classes)
ax2.set_ylabel('Number of Images')
ax2.set_title('Large Loss Weights Adjustment')
ax2.grid(linestyle='--', alpha=0.5)

yellow_patch = mpatches.Patch(color='y', label='Total Images')
green_patch = mpatches.Patch(color='g', label='Correctly Classified Images')
ax1.legend(handles=[yellow_patch, green_patch], loc='lower center', bbox_to_anchor=(0.25, -0.12), ncol=2)

plt.show()
