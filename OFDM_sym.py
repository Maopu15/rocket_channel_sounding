import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from mpl_toolkits import mplot3d


class OFDM_sym(object):

    def __init__(self,snr_dB):
        self.K = 512 # number of OFDM subcarriers
        self.CP = self.K//8  # length of the cyclic prefix: 25% of the block
        self.P = 63 # number of pilot carriers per OFDM block
        self.symPerFrame = 5

        self.pilotValue = 7+7j # The known value each pilot transmits
        self.allCarriers = np.arange(self.K)  # indices of all subcarriers ([0, 1, ... K-1])
        self.pilotCarriers = self.allCarriers[::self.K//self.P] # Pilots is every (K/P)th carrier.
        # For convenience of channel estimation, let's make the last carriers also be a pilot
        self.pilotCarriers = np.hstack([self.pilotCarriers, np.array([self.allCarriers[-1]])])
        self.P = self.P+1
        # data carriers are all remaining carriers
        self.dataCarriers = np.delete(self.allCarriers, self.pilotCarriers)

        self.mu = 6  # bits per symbol (i.e. 16QAM)
        self.payloadBits_per_OFDM = len(self.dataCarriers)*self.mu  # number of payload bits per OFDM symbol

        self.mapping_table = self.mapping_table()
        self.demapping_table = {v: k for k, v in self.mapping_table.items()}

        tap_delays = [0, 1, 3, 5]  # in samples
        tap_weights = [1, 0.6, 0.3, 0.1+1j]
        self.channelResponse = self.create_multipath_fir_filter(tap_delays, tap_weights)
        self.H_exact = np.fft.fft(self.channelResponse, self.K)

        self.SNRdb = snr_dB  # signal to noise-ratio in dB at the receiver
        self.H_est = None
        pass

    def execute(self,cfo,sto):



        self.OFDM_preamble = self.ofdm_preamble_create()

        #Create packet
        OFDM_TX,bits = self.ofdm_symbol_create()
        OFDM_TX =np.hstack([self.OFDM_preamble, OFDM_TX])
        OFDM_TX=self.addSTO(OFDM_TX,sto)
        OFDM_TX = self.addCFO(OFDM_TX,cfo)
        OFDM_RX = self.addChannel(OFDM_TX)
        #self.plotRXTX(OFDM_RX, OFDM_TX)
        data_start_index = self.findPreamble(OFDM_RX)+len(self.OFDM_preamble)
        #data_start_index = self.findPreamble(OFDM_RX)
        #OFDM_ch =OFDM_RX[:data_start_index+len(self.preamble)]
        #OFDM_ch = self.removeCP(OFDM_ch)
        #OFDM_ch = self.DFT(OFDM_ch)
        #self.plotRXTX(OFDM_RX, OFDM_TX)
        OFDM_RX = OFDM_RX[data_start_index:data_start_index+self.K+self.CP]
        #self.plotRXTX(OFDM_ch, OFDM_TX[:data_start_index+len(self.preamble)])
        OFDM_RX_noCP = self.removeCP(OFDM_RX)
        OFDM_demod = self.DFT(OFDM_RX_noCP)
        print(len(OFDM_demod))

        Hest = self.channel_estimate(OFDM_demod)
        #Hest =self.channel_estimate_preamble(OFDM_ch)
        equalized_Hest = self.equalize(OFDM_demod, Hest)
        QAM_est = self.get_payload(equalized_Hest)
        #plt.plot(QAM_est.real, QAM_est.imag, 'bo')
        #plt.show()

        PS_est, hardDecision = self.demapping(QAM_est)

        #for qam, hard in zip(QAM_est, hardDecision):
        #    plt.plot([qam.real, hard.real], [qam.imag, hard.imag], 'b-o');
        #    plt.plot(hardDecision.real, hardDecision.imag, 'ro')
        #plt.show()
        bits_est = self.PS(PS_est)
        #self.plotRXTX(OFDM_RX_noCP, OFDM_TX[data_start_index:data_start_index+self.K])
        return np.sum(np.abs(bits-bits_est)),len(bits)
        pass

    def SP(self, bits):
        return bits.reshape((len(self.dataCarriers), self.mu))

    def PS(self,bits):
        return bits.reshape((-1,))

    def mapping_table(self):
        return {
        (0, 0, 0, 0, 0, 0): -7 - 7j,
        (0, 0, 0, 0, 0, 1): -7 - 5j,
        (0, 0, 0, 0, 1, 0): -7 - 1j,
        (0, 0, 0, 0, 1, 1): -7 - 3j,
        (0, 0, 0, 1, 0, 0): -7 + 7j,
        (0, 0, 0, 1, 0, 1): -7 + 5j,
        (0, 0, 0, 1, 1, 0): -7 + 1j,
        (0, 0, 0, 1, 1, 1): -7 + 3j,
        (0, 0, 1, 0, 0, 0): -5 - 7j,
        (0, 0, 1, 0, 0, 1): -5 - 5j,
        (0, 0, 1, 0, 1, 0): -5 - 1j,
        (0, 0, 1, 0, 1, 1): -5 - 3j,
        (0, 0, 1, 1, 0, 0): -5 + 7j,
        (0, 0, 1, 1, 0, 1): -5 + 5j,
        (0, 0, 1, 1, 1, 0): -5 + 1j,
        (0, 0, 1, 1, 1, 1): -5 + 3j,
        (0, 1, 0, 0, 0, 0): -1 - 7j,
        (0, 1, 0, 0, 0, 1): -1 - 5j,
        (0, 1, 0, 0, 1, 0): -1 - 1j,
        (0, 1, 0, 0, 1, 1): -1 - 3j,
        (0, 1, 0, 1, 0, 0): -1 + 7j,
        (0, 1, 0, 1, 0, 1): -1 + 5j,
        (0, 1, 0, 1, 1, 0): -1 + 1j,
        (0, 1, 0, 1, 1, 1): -1 + 3j,
        (0, 1, 1, 0, 0, 0): -3 - 7j,
        (0, 1, 1, 0, 0, 1): -3 - 5j,
        (0, 1, 1, 0, 1, 0): -3 - 1j,
        (0, 1, 1, 0, 1, 1): -3 - 3j,
        (0, 1, 1, 1, 0, 0): -3 + 7j,
        (0, 1, 1, 1, 0, 1): -3 + 5j,
        (0, 1, 1, 1, 1, 0): -3 + 1j,
        (0, 1, 1, 1, 1, 1): -3 + 3j,
        (1, 0, 0, 0, 0, 0): 7 - 7j,
        (1, 0, 0, 0, 0, 1): 7 - 5j,
        (1, 0, 0, 0, 1, 0): 7 - 1j,
        (1, 0, 0, 0, 1, 1): 7 - 3j,
        (1, 0, 0, 1, 0, 0): 7 + 7j,
        (1, 0, 0, 1, 0, 1): 7 + 5j,
        (1, 0, 0, 1, 1, 0): 7 + 1j,
        (1, 0, 0, 1, 1, 1): 7 + 3j,
        (1, 0, 1, 0, 0, 0): 5 - 7j,
        (1, 0, 1, 0, 0, 1): 5 - 5j,
        (1, 0, 1, 0, 1, 0): 5 - 1j,
        (1, 0, 1, 0, 1, 1): 5 - 3j,
        (1, 0, 1, 1, 0, 0): 5 + 7j,
        (1, 0, 1, 1, 0, 1): 5 + 5j,
        (1, 0, 1, 1, 1, 0): 5 + 1j,
        (1, 0, 1, 1, 1, 1): 5 + 3j,
        (1, 1, 0, 0, 0, 0): 1 - 7j,
        (1, 1, 0, 0, 0, 1): 1 - 5j,
        (1, 1, 0, 0, 1, 0): 1 - 1j,
        (1, 1, 0, 0, 1, 1): 1 - 3j,
        (1, 1, 0, 1, 0, 0): 1 + 7j,
        (1, 1, 0, 1, 0, 1): 1 + 5j,
        (1, 1, 0, 1, 1, 0): 1 + 1j,
        (1, 1, 0, 1, 1, 1): 1 + 3j,
        (1, 1, 1, 0, 0, 0): 3 - 7j,
        (1, 1, 1, 0, 0, 1): 3 - 5j,
        (1, 1, 1, 0, 1, 0): 3 - 1j,
        (1, 1, 1, 0, 1, 1): 3 - 3j,
        (1, 1, 1, 1, 0, 0): 3 + 7j,
        (1, 1, 1, 1, 0, 1): 3 + 5j,
        (1, 1, 1, 1, 1, 0): 3 + 1j,
        (1, 1, 1, 1, 1, 1): 3 + 3j
    }

    def createFilter(self,):
        return None

    def bitMapping(self,bits):
        return np.array([self.mapping_table[tuple(b)] for b in bits])

    def OFDM_symbol(self,QAM_payload):
        symbol = np.zeros(self.K, dtype=complex) # the overall K subcarriers
        symbol[self.pilotCarriers] = self.pilotValue  # allocate the pilot subcarriers
        symbol[self.dataCarriers] = QAM_payload  # allocate the pilot subcarriers
        return symbol

    def IDFT(self,OFDM_data):
        return np.fft.ifft(OFDM_data)

    def DFT(self,OFDM_RX):
        return np.fft.fft(OFDM_RX)

    def addCP(self,OFDM_time):
        cp = OFDM_time[-self.CP:]
        return np.hstack([cp, OFDM_time])

    def removeCP(self,signal):
        return signal[self.CP:(self.CP+self.K)]
    def addChannel(self,signal):
        convolved = np.convolve(signal, self.channelResponse)
        signal_power = np.mean(abs(convolved**2))
        sigma2 = signal_power * 10**(-self.SNRdb/10) # noise power based on SNR

        print("RX Signal power: %.4f. Noise power: %.4f" % (signal_power, sigma2))

        # Generate complex noise
        noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
        return convolved + noise

    def channel_estimate(self,OFDM_demod):
        pilots = OFDM_demod[self.pilotCarriers]  # Pilot values from RX demodulated signal
        Hest_at_pilots = pilots/self.pilotValue  # divide by the transmitted pilot values

        # Interpolation between the pilot carriers, absolute value and phase separately
        Hest_abs = scipy.interpolate.interp1d(self.pilotCarriers, abs(Hest_at_pilots), kind='linear')(self.allCarriers)
        Hest_phase = scipy.interpolate.interp1d(self.pilotCarriers, np.angle(Hest_at_pilots), kind='linear')(self.allCarriers)
        Hest = Hest_abs * np.exp(1j * Hest_phase)
        self.H_est = Hest
        return Hest

    def plotRXTX(self, rx_sig, tx_sig):
        plt.figure(figsize=(8, 2))
        plt.plot(abs(tx_sig), label='TX signal')
        plt.plot(abs(rx_sig), label='RX signal')
        plt.legend(fontsize=10)
        plt.xlabel('Time')
        plt.ylabel('$|x(t)|$')
        plt.grid(True)
        plt.show()

    def equalize(self, OFDM_demod, Hest):
        return OFDM_demod/Hest

    def get_payload(self, equalized):
        return equalized[self.dataCarriers]

    def demapping(self, QAM):
        # array of possible constellation points
        constellation = np.array([x for x in self.demapping_table.keys()])

        # calculate distance of each RX point to each possible point
        dists = abs(QAM.reshape((-1, 1)) - constellation.reshape((1, -1)))

        # for each element in QAM, choose the index in constellation
        # that belongs to the nearest constellation point
        const_index = dists.argmin(axis=1)

        # get back the real constellation point
        hardDecision = constellation[const_index]

        # transform the constellation point into the bit groups
        return np.vstack([self.demapping_table[C] for C in hardDecision]), hardDecision

    def random_qam(self):
        qam = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
        return np.random.choice(qam, size=(self.K), replace=True)

    def ofdm_modulate(self, qam):
        assert (len(qam) == self.K)
        fd_data = np.zeros(self.K, dtype=complex)
        fd_data= qam  # modulate in the center of the frequency
        fd_data = np.fft.fftshift(fd_data)
        symbol = np.fft.ifft(fd_data) * np.sqrt(self.K)
        return np.hstack([symbol[-self.CP:], symbol])

    def ofdm_symbol_create(self):

        #bits = np.random.binomial(n=1, p=0.5, size=(self.payloadBits_per_OFDM//3, ))
        #bits = np.repeat(bits,3)
        bits = np.random.binomial(n=1, p=0.5, size=(self.payloadBits_per_OFDM, ))
        bits_SP = self.SP(bits)
        QAM = self.bitMapping(bits_SP)
        OFDM_data = self.OFDM_symbol(QAM)
        OFDM_time = self.IDFT(OFDM_data)
        OFDM_withCP = self.addCP(OFDM_time)
        return OFDM_withCP,bits

    def ofdm_preamble_create(self):

        bits = np.random.binomial(n=1, p=0.5, size=(self.K*6, ))
        #bits_SP = self.SP(bits)
        bits_SP = bits.reshape((self.K, self.mu))
        QAM = self.bitMapping(bits_SP)
        self.preamble = QAM
        #OFDM_data = self.OFDM_symbol(QAM)
        OFDM_data = QAM
        OFDM_data[::2] = 0
        OFDM_data = OFDM_data*2
        OFDM_time = self.IDFT(OFDM_data)
        OFDM_withCP = self.addCP(OFDM_time)
        return OFDM_withCP

    def addCFO(self,signal, cfo):  # Add carrier frequency offset
        return signal * np.exp(2j * np.pi * cfo * np.arange(len(signal)))

    def addSTO(self,signal, sto):  # add some time offset
        return np.hstack([np.zeros(sto), signal])

    def findPreamble(self, received_signal):
        # Calculate the cross-correlation between the received signal and the known preamble
        correlation = np.correlate(received_signal, self.OFDM_preamble, mode='valid')

        # Find the index of the maximum correlation value
        max_correlation_index = np.argmax(correlation)

        # Optional: Plot the correlation for visualization
        plt.plot(correlation)
        plt.title('Cross-Correlation with Preamble')
        plt.xlabel('Sample index')
        plt.ylabel('Correlation value')
        plt.show()

        return max_correlation_index

    def create_multipath_fir_filter(self,tap_delays, tap_weights):
        """
        Creates a multipath FIR filter to simulate a channel.

        Parameters:
        tap_delays (list): List of tap delays in samples.
        tap_weights (list): List of tap weights (complex or real).

        Returns:
        np.array: The impulse response of the channel.
        """
        # Create an array of zeros with a length equal to the last tap delay
        max_delay = max(tap_delays)
        h = np.zeros(max_delay + 1, dtype=np.complex64)

        # Assign the weights to the corresponding tap delays
        for tap_delay, weight in zip(tap_delays, tap_weights):
            h[tap_delay] = weight

        return h

    def channel_estimate_preamble(self,OFDM_demod):
        Hest_at_preamble = (OFDM_demod)/(self.preamble[self.CP:])  # divide by the transmitted pilot values
        self.H_est = np.hstack([Hest_at_preamble[-self.CP:],Hest_at_preamble])
        return np.hstack([Hest_at_preamble[-self.CP:],Hest_at_preamble])

    def ofdm_symbols_create(self, num_symbols):
        all_OFDM_symbols = []
        all_bits = []
        for _ in range(num_symbols):
            OFDM_symbol, bits = self.ofdm_symbol_create()
            all_OFDM_symbols.append(OFDM_symbol)
            all_bits.append(bits)
        return np.concatenate(all_OFDM_symbols), np.concatenate(all_bits)