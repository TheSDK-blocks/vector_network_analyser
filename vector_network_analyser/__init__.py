"""
=========
vector_network_analyser
=========

vector_network_analyser model template The System Development Kit
Used as a template for all TheSyDeKick Entities.

Current docstring documentation style is Numpy
https://numpydoc.readthedocs.io/en/latest/format.html

This text here is to remind you that documentation is important.
However, youu may find it out the even the documentation of this 
entity may be outdated and incomplete. Regardless of that, every day 
and in every way we are getting better and better :).

Initially written by Marko Kosunen, marko.kosunen@aalto.fi, 2017.

"""

import os
import sys
import itertools
if not (os.path.abspath('../../thesdk') in sys.path):
    sys.path.append(os.path.abspath('../../thesdk'))

from thesdk import *
import skrf as rf
from matplotlib import pyplot as plt

import numpy as np

class vector_network_analyser(thesdk):
    """
    Vector Network Analyser entity for plotting S-parameters from a Touchstone file.
    This entity uses skikit-RF package to plot the S-parameters in varying formats
    (see self.supported_plots for support plot formats).

    Attributes
    ----------
    supported_plots: list[str]
        List of supported plot types. This is directly taken from the skikit-RF documentation.
    plot_type: str
        Type of plot. Must be found in self.supported_plots.
        Default: 'db'. 
    plot_indices: list[tuple[int]]
        Plot the S-parameters for listed indices. Empty list plot all
        available parameters.
        Default: []
    sparam_file: str or PathObject
        Path to the Touchstone file containing the S-parameters
    num_ports: int
        Number of ports for the network. Inferred from self.sparam_file extension.
        Do not set manually.
    xlim: tuple[int]
        A two element tuple for setting x limits of the plot
        Setting to None scales the limits to the data.
        Default: None
    ylim: tuple[int]
        A two element tuple for setting y limits of the plot
        Setting to None scales the limits to the data.
        Default: None
    """
    @property
    def _classfile(self):
        return os.path.dirname(os.path.realpath(__file__)) + "/"+__name__

    def __init__(self,*arg): 
        self.print_log(type='I', msg='Inititalizing %s' %(__name__)) 
        self.proplist = [ 'plot_type', 'plot_indices', 'sparam_file' ]    # Properties that can be propagated from parent
        self.model='py';             # Can be set externally, but is not propagated
        self.par= False              # By default, no parallel processing
        self.queue= []               # By default, no parallel processing

        self.plot_indices = []
        self.plot_type = 'db'
        self.sparam_file = None
        self.ylabel = ''
        self.xlabel = ''
        self.xlim = None
        self.ylim = None

        if len(arg)>=1:
            parent=arg[0]
            self.copy_propval(parent,self.proplist)
            self.parent =parent;

    @property
    def supported_plots(self):
        """
        List of supported plot types. Specify desired plot by setting
        self.plot_type
        """
        if not hasattr(self, '_supported_plots'):
            self._supported_plots=['smith', 'db', 'mag', 're', 'im', 'deg', 'deg_unwrap']
        return self._supported_plots

    @supported_plots.setter
    def supported_plots(self, val):
        self._supported_plots=val


    def enumerate_port_indices(self):
        m = [i+1 for i in range(self.num_ports)]
        n = [i+1 for i in range(self.num_ports)]
        self.available_port_indices = list(itertools.product(m,n))

    def float_to_si_string(self, num, precision=6):
        """Converts the given floating point number to a SI prefix string and divider.

        Parameters
        ----------
        num : float
            the number to convert.
        precision : int
            number of significant digits, defaults to 6.

        Returns
        -------
        x_scale : str
            the SI string of the value.
        x_scaler : float
            the scaler (divider) that can be used to normalize the signal to the
            given SI unit.
        """
        si_mag = [-18, -15, -12, -9, -6, -3, 0, 3, 6, 9, 12]
        si_pre = ['a', 'f', 'p', 'n', 'u', 'm', '', 'k', 'M', 'G', 'T']

        if abs(num) < 1e-21:
            return '',1
        exp = np.log10(abs(num))

        pre_idx = len(si_mag) - 1
        for idx in range(len(si_mag)):
            if exp < si_mag[idx]:
                pre_idx = idx - 1
                break

        res = 10.0 ** (si_mag[pre_idx])
        return si_pre[pre_idx],res

    def get_port_label(self, idx):
        if idx in self.available_port_indices:
            return f'S{idx[0]}{idx[1]}'
        else:
            self.print_log(type='F', msg='S{idx[0]}{idx[1]} does not exist for a {self.num_ports} port network!')

    def plot(self):
        '''Guideline. Isolate python processing to main method.
        
        To isolate the interna processing from IO connection assigments, 
        The procedure to follow is
        1) Assign input data from input to local variable
        2) Do the processing
        3) Assign local variable to output

        '''
        fig,ax=plt.subplots()
        # Call plot function
        try:
            getattr(self.network, f'plot_s_{self.plot_type}')()
        except AttributeError:
            if self.plot_type in self.supported_plots:
                self.print_log(type='F', msg=f'Something went terribly wrong. {self.plot_type} is in self.supported_plots, but Network object has no such attribute!')
            else:
                self.print_log(type='F', msg=f'Invalid plot_type: {self.plot_type}')
        self.data = {}
        for line in ax.lines:
            label = line.get_label().split(',')[1][1:]
            x,y = line.get_data()
            self.data[label] = np.vstack((x,y)).T
        # Now we have obtained the data. Close the old figure, and draw a better one
        plt.close(fig)
        fig,ax=plt.subplots()
        # Assume all data have the same frequency vector
        x_scale,x_scaler=self.float_to_si_string(np.max(x))
        freq = self.data[label][:,0] / x_scaler
        if len(self.plot_indices) == 0: # Plot everthing
            plot_keys = [self.get_port_label(idx) for idx in self.available_port_indices]
        else:
            plot_keys = [self.get_port_label(idx) for idx in self.plot_indices]
        for key in plot_keys:
            try:
                ax.plot(freq, self.data[key][:,1], label=key)
            except KeyError:
                self.print_log(type='W', msg=f'{key} not found in Touchstone file!!')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_xlabel(f'Frequency ({x_scale}Hz)')
        if self.xlim:
            ax.set_xlim(*self.xlim)
        else:
            ax.set_xlim(min(freq), max(freq))
        if self.ylim:
            ax.set_ylim(*self.ylim)
        plt.legend()
        plt.show(block=False)

    def check_input_file(self):
        match = re.search('s([0-9])+p', self.sparam_file)
        if not match:
            self.print_log(type='F', msg='Incorrect file format! Currently, only touchstone is supported!')
        if not os.path.isfile(self.sparam_file):
            self.print_log(type='F', msg=f'File {self.sparam_file} does not exist!')
        self.num_ports = int(match.group(1))
        self.print_log(msg=f'Inferred number of ports from file extension: {self.num_ports}')
        self.network = rf.Network(self.sparam_file)

    def main(self):
        self.check_input_file()
        self.enumerate_port_indices()
        self.plot()

    def run(self,*arg):
        '''Guideline: Define model depencies of executions in `run` method.

        '''
        if len(arg)>0:
            self.par=True      #flag for parallel processing
            self.queue=arg[0]  #multiprocessing.queue as the first argument
        if self.model=='py':
            self.main()

if __name__=="__main__":
    import matplotlib.pyplot as plt
    from  vector_network_analyser import *
    from  vector_network_analyser.controller import controller as vector_network_analyser_controller
    import plot_format
    plot_format.set_style('isscc')
    import pdb
    import math

    sparam_file = '../test_data/transformer.s4p'
    dut = vector_network_analyser()
    dut.sparam_file = sparam_file
    dut.plot_indices = [(1,1), (2,1)]
    dut.plot_type='db'
    dut.run()
    input()

