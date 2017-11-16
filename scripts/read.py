#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is used to manipulate and visualize the output chains from the JAGS
MCMC code.

"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
import matplotlib.pyplot as plt
import corner
from scipy.stats import binned_statistic_2d


# The Posterior class
class Posterior(object):
    """
    Read and analyze the posterior distribution given the paths of the output
    files of an MCMC simulation.

    Args:
        path_index (`str`): Path of the index file.
        path_chain (`str`): Path of the chain file.
    """
    def __init__(self, path_index, path_chain):

        # Read the index file
        self.var = np.loadtxt(path_index, usecols=(0,), dtype=str)
        self.limit = np.loadtxt(path_index, usecols=(1, 2), dtype=int)
        self.limit += (-1)  # Python index starts with 0 instead of 1
        n_var = len(self.var)  # Number of variables

        # Read the chain file
        self.long_chain = np.loadtxt(path_chain, usecols=(1,))
        self.steps = np.loadtxt(path_chain, usecols=(0,))

        # The chain library
        self.chain = {}
        self.rank = {}

        for i in range(n_var):
            self.chain[self.var[i]] = self.long_chain[
                                      self.limit[i, 0]:self.limit[i, 1] + 1]
            self.rank[self.var[i]] = self.steps[
                                     self.limit[i, 0]:self.limit[i, 1] + 1]

    # Print some simple statistics
    def print_stats(self, variable, percentile):
        """
        Print the mean, standard deviation and a specific percentile of a
        specific variable in the chain.

        Args:
            variable (`str`): Variable name.
            percentile (`int`): Percentile.
        """
        mean = np.mean(self.chain[variable])
        std = np.std(self.chain[variable])
        perc = np.percentile(self.chain[variable], q=percentile)

        # And then just print them
        print('Mean of `%s` = %f' % (variable, mean))
        print('Standard deviation of `%s` = %f' % (variable, std))
        print('Percentile %i of `%s` = %f' % (percentile, variable, perc))

    # Make a trace plot
    def plot_trace(self, variable, output=None, x_range=None, y_range=None):
        """
        Plot the trace of a specific variable in the chain.

        Args:
            variable (`str`): Variable name.
            output (`str`, optional): Plot output file name.
            x_range (`tuple`, optional): Range of values for the x-axis.
            y_range (`tuple`, optional): Range of values for the y-axis.
        """
        plt.plot(self.rank[variable], self.chain[variable], lw=1)
        plt.ylabel(variable)
        plt.xlim(x_range)
        plt.ylim(y_range)

        if output is None:
            plt.show()
        else:
            plt.savefig(output)
            plt.close()

    # Compute the histogram of a variable and optionally plot it
    def compute_hist(self, variable, n_bins=None, plot=False, output=None):
        """
        Compute the histogram of a specific variable in the chain and optionally
        plot it.

        Args:
            variable (`str`): Variable name.
            n_bins (`int`, optional): Number of bins for the histogram.
            plot (`bool`, optional): If True, plot the histogram.
            output (`str`, optional): Path to the output histogram plot.

        Returns:
            hist (`numpy.array`): Histogram.
            bins (`numpy.array`): Bins of the histogram.
        """
        fig, ax = plt.subplots()
        hist, bins, patches = ax.hist(self.chain[variable], bins=n_bins,
                                      normed=True, histtype='step')

        if plot is True:
            if output is None:
                plt.show()
            else:
                plt.savefig(output)
                plt.close()
        else:
            plt.close()

        return hist, bins

    # Plot contours
    def plot_contour(self):
        """
        Plot the corner plot (which includes the contours and marginalized
        posteriors of the chains.
        """
        data = []
        label = []
        for key in self.chain.keys():
            data.append(self.chain[key])
            label.append(key)
        data = np.array(data).T

        # Plot corner
        corner.corner(data, labels=label, levels=(0.68, 0.95),
                      quantiles=[0.16, 0.84], show_titles=True,
                      title_kwargs={"fontsize": 12})
        plt.tight_layout()
        plt.show()

    # Compute the means of binned data
    def binned_data(self, var1, var2, n_bins=10):
        """
        Computes the expected values of variable 2 (`y`) given binned values of
        variable 1 (`x`) and vice-versa.

        Args:
            var1 (`str`): Variable 1 name.
            var2 (`str`): Variable 2 name.
            n_bins (`int`, optional): Number of bins.

        Returns:
            x (`numpy.array`): Binned values of variable 1.
            y_given_x (`numpy.array`): Expected values of variable 2 given 1.
            y (`numpy.array`): Binned values of variable 2.
            x_given_y (`numpy.array`): Expected values of variable 1 given 2.
        """
        xs = self.chain[var1]
        ys = self.chain[var2]
        count, bin_x, bin_y, bin_number = binned_statistic_2d(xs, ys, None,
                                                              'count',
                                                              bins=n_bins)
        count_y = np.sum(count, axis=(1,))
        count_x = np.sum(count, axis=(0,))
        inds_x = np.digitize(xs, bin_x, right=True)
        inds_y = np.digitize(ys, bin_y, right=True)
        sum_y = np.zeros(n_bins, float)
        sum_x = np.zeros(n_bins, float)
        for i in range(len(xs)):
            sum_y[inds_x[i] - 1] += ys[i]
            sum_x[inds_y[i] - 1] += xs[i]
        y_given_x = sum_y / count_y
        x_given_y = sum_x / count_x

        x = np.array([(bin_x[i + 1] + bin_x[i]) / 2
                       for i in range(len(bin_x) - 1)])
        y = np.array([(bin_y[i + 1] + bin_y[i]) / 2
                       for i in range(len(bin_y) - 1)])

        return x, y_given_x, y, x_given_y


# Test
if __name__ == '__main__':
    _path_index = '../CODAindex_fakesampleregr.txt'
    _path_chain = '../CODAchain_fakesampleregr.txt'
    chain = Posterior(_path_index, _path_chain)

    #chain.print_stats('x', 95)
    #chain.plot_trace('s', x_range=(0, 5000))
    #_hist, _bins = chain.compute_hist('s', n_bins=100, plot=True)
    #chain.plot_contour()
    chain.binned_data('obsx[1]', 'obsx[2]', 5)
