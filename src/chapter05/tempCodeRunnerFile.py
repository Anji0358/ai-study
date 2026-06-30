
# plt.ylabel('Explained variance ratio')
# plt.xlabel('Principal component index')
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()

eigen_pairs=[(np.abs(eigen_vals[i]),eigen_vacs[:,i])
             for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k:k[0],reverse=True)
