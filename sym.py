import numpy as np
import matplotlib.pyplot as plt
from OFDM_sym import OFDM_sym

# Define ranges for SNR, CFO, and STO
snr_range = [100]
cfo_range = [0.0002]  # Example values for carrier frequency offset
sto_range = [0]  # Example values for sample timing offset
# Initialize a list to store the results and BER values
all_signals = []
ber_values_snr = {snr: [] for snr in snr_range}  # Dictionary to store BER values for each SNR



# Store the exact channel (only need to store once)
H_exact = None

for SNR_dB in snr_range:
    for cfo in cfo_range:
        for sto in sto_range:
            H_est_results = []
            ber_values = []  # Store BER values for each combination
            for instance in range(10):
                ofdm_instance = OFDM_sym(SNR_dB)
                num_bit_errors, total_bits = ofdm_instance.execute(cfo, sto)  # Include CFO and STO
                ber = num_bit_errors / total_bits
                ber_values.append(ber)
                H_est_results.append(ofdm_instance.H_est)
                if H_exact is None:
                    H_exact = ofdm_instance.H_exact

            avg_H_est = np.mean(H_est_results, axis=0)
            all_signals.append((SNR_dB, cfo, sto, avg_H_est))
            ber_values_snr[SNR_dB].extend(ber_values)  # Extend the BER list for the current SNR

subcarrier_numbers = np.arange(len(H_exact))

# Plotting the channel estimates for different SNR levels
plt.figure(figsize=(15, 8))
for snr in snr_range:
    filtered_signals = [item for item in all_signals if item[0] == snr and item[1] == 0 and item[2] == 0]
    for _, _, _, h_est in filtered_signals:
        plt.plot(subcarrier_numbers, np.abs(h_est), label=f'SNR={snr}')
        for pilot in ofdm_instance.pilotCarriers:
            plt.axvline(x=pilot, color='gray', linestyle='--', linewidth=1)  # Marking the pilot carriers

plt.plot(subcarrier_numbers, np.abs(H_exact), 'k--', linewidth=2, label='H_exact')
plt.title('Channel Estimates for Different SNR Levels')
plt.xlabel('Subcarrier Number')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)
plt.show()

# Plotting BER vs. SNR
plt.figure(figsize=(10, 6))
for snr in snr_range:
    avg_ber = np.mean(ber_values_snr[snr])
    plt.plot(snr, avg_ber, 'o', label=f'SNR = {snr} dB')

plt.title('BER vs SNR for OFDM System')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.yscale('log')
plt.ylim(1e-8)
plt.grid(True)
plt.legend()
plt.show()

# Plotting the channel estimates for different CFO levels
plt.figure(figsize=(15, 8))
for cfo in cfo_range:
    filtered_signals = [item for item in all_signals if item[1] == cfo and item[0] == 100]  # Filter for a specific SNR, for instance, 20 dB
    for _, _, _, h_est in filtered_signals:
        plt.plot(subcarrier_numbers, np.abs(h_est), label=f'CFO={cfo} Hz')

plt.plot(subcarrier_numbers, np.abs(H_exact), 'k--', linewidth=2, label='H_exact')
plt.title('Channel Estimates for Different CFO Levels')
plt.xlabel('Subcarrier Number')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)
plt.show()