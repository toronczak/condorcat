import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from scipy import special
from scipy.stats import qmc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading


class CondorcetSimulation:
    def __init__(self, root):
        """Initialize the Condorcet Simulation GUI."""
        self.root = root
        self.root.title("condorcat – a Condorcet Jury Theorem simulator")
        self.root.geometry("650x810")
        self.root.minsize(650, 810)

        # Configure grid layout
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Initialize variables
        self.n_voters = tk.IntVar(value=5)
        self.p = tk.DoubleVar(value=0.6)
        self.advanced_mode = tk.BooleanVar(value=False)
        self.simulation_runs = tk.IntVar(value=128)  # Changed to power of 2 for LHS
        self.p_min_threshold = tk.DoubleVar(value=0.3)
        self.p_max_threshold = tk.DoubleVar(value=0.7)
        self.is_closing = False

        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="NSEW")
        self.main_frame.columnconfigure(0, weight=1)

        # Create UI components
        self.create_widgets()

        # Placeholder for the plot canvas
        self.plot_canvas = None

        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Lock for threading
        self.lock = threading.Lock()

    def create_widgets(self):
        """Create and arrange widgets in the main frame."""
        # Input Frame
        input_frame = ttk.LabelFrame(self.main_frame, text="Inputs", padding="10")
        input_frame.grid(row=0, column=0, sticky="EW", padx=5, pady=5)
        input_frame.columnconfigure(1, weight=1)

        # Number of Voters
        ttk.Label(input_frame, text="Number of Voters (n):").grid(row=0, column=0, sticky="W", pady=2)
        voters_spinbox = ttk.Spinbox(
            input_frame, from_=2, to=10000, textvariable=self.n_voters, width=10,
            validate='all', validatecommand=(self.root.register(self.validate_n_spinbox), '%P')
        )
        voters_spinbox.grid(row=0, column=1, sticky="EW", pady=2)

        # Advanced Mode Checkbox
        advanced_checkbox = ttk.Checkbutton(
            input_frame, text="Advanced Mode: Distributed Voter Accuracies (LHS)",
            variable=self.advanced_mode, command=self.toggle_mode
        )
        advanced_checkbox.grid(row=1, column=0, columnspan=2, sticky="W", pady=2)

        # Voter Accuracy (p) Slider (Basic Mode)
        self.p_frame = ttk.Frame(input_frame)
        self.p_frame.grid(row=2, column=0, columnspan=2, sticky="EW", pady=2)
        self.p_frame.columnconfigure(1, weight=1)

        ttk.Label(self.p_frame, text="Voter Accuracy (p):").grid(row=0, column=0, sticky="W")
        self.p_scale = ttk.Scale(
            self.p_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
            variable=self.p, command=self.update_p_label
        )
        self.p_scale.grid(row=0, column=1, sticky="EW", padx=5)
        self.p_display = ttk.Label(self.p_frame, text=f"{self.p.get():.2f}")
        self.p_display.grid(row=0, column=2, sticky="W")

        # Advanced Mode Parameters
        self.advanced_frame = ttk.Frame(input_frame)
        self.advanced_frame.grid(row=3, column=0, columnspan=2, sticky="EW", pady=2)
        self.advanced_frame.columnconfigure(1, weight=1)

        # Simulation Runs (powers of 2 for LHS)
        ttk.Label(self.advanced_frame, text="Simulation Runs (2^n):").grid(row=0, column=0, sticky="W", pady=2)
        sim_runs_values = [str(2**n) for n in range(7, 13)]  # 128 to 8192
        sim_runs_combobox = ttk.Combobox(
            self.advanced_frame, values=sim_runs_values, 
            textvariable=self.simulation_runs, width=10, state="readonly"
        )
        sim_runs_combobox.grid(row=0, column=1, sticky="EW", pady=2)
        if str(self.simulation_runs.get()) not in sim_runs_values:
            sim_runs_combobox.set(sim_runs_values[0])

        # Lower Threshold
        ttk.Label(self.advanced_frame, text="Voter Accuracy Lower Threshold (p_min):").grid(row=1, column=0, sticky="W", pady=2)
        self.p_min_scale = ttk.Scale(
            self.advanced_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
            variable=self.p_min_threshold, command=self.update_min_threshold_label
        )
        self.p_min_scale.grid(row=1, column=1, sticky="EW", padx=5)
        self.p_min_display = ttk.Label(self.advanced_frame, text=f"{self.p_min_threshold.get():.2f}")
        self.p_min_display.grid(row=1, column=2, sticky="W")

        # Upper Threshold
        ttk.Label(self.advanced_frame, text="Voter Accuracy Upper Threshold (p_max):").grid(row=2, column=0, sticky="W", pady=2)
        self.p_max_scale = ttk.Scale(
            self.advanced_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
            variable=self.p_max_threshold, command=self.update_max_threshold_label
        )
        self.p_max_scale.grid(row=2, column=1, sticky="EW", padx=5)
        self.p_max_display = ttk.Label(self.advanced_frame, text=f"{self.p_max_threshold.get():.2f}")
        self.p_max_display.grid(row=2, column=2, sticky="W")

        # Initially hide advanced parameters
        self.advanced_frame.grid_remove()

        # Run Simulation Button
        self.run_button = ttk.Button(
            self.main_frame, text="Run Simulation", command=self.run_simulation
        )
        self.run_button.grid(row=1, column=0, pady=10)

        # Results Frame
        results_frame = ttk.LabelFrame(self.main_frame, text="Results", padding="10")
        results_frame.grid(row=2, column=0, sticky="EW", padx=5, pady=5)
        results_frame.columnconfigure(0, weight=1)

        ttk.Label(results_frame, text="Probability Majority is Correct:").grid(row=0, column=0, sticky="W")
        self.result_value = ttk.Label(
            results_frame, text="", foreground="blue", font=("Arial", 10, "bold")
        )
        self.result_value.grid(row=1, column=0, sticky="W", pady=(5, 0))

        # Plot Frame
        plot_frame_container = ttk.LabelFrame(self.main_frame, text="Plot", padding="10")
        plot_frame_container.grid(row=3, column=0, sticky="NSEW", padx=5, pady=5)
        plot_frame_container.columnconfigure(0, weight=1)
        plot_frame_container.rowconfigure(0, weight=1)

        self.plot_frame = ttk.Frame(plot_frame_container)
        self.plot_frame.grid(row=0, column=0, sticky="NSEW")
        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)

        # Progress Bar
        self.progress = ttk.Progressbar(
            self.main_frame, orient='horizontal', mode='indeterminate'
        )
        self.progress.grid(row=4, column=0, sticky="EW", padx=5, pady=5)
        self.progress.grid_remove()

    def toggle_mode(self):
        """Toggle between Basic and Advanced modes."""
        if self.advanced_mode.get():
            self.p_frame.grid_remove()
            self.advanced_frame.grid()
        else:
            self.advanced_frame.grid_remove()
            self.p_frame.grid()

    def update_p_label(self, event=None):
        """Update the display label for probability p."""
        if not self.advanced_mode.get():
            self.p_display.config(text=f"{self.p.get():.2f}")

    def update_min_threshold_label(self, event=None):
        """Update the display label for voter accuracy lower threshold."""
        if self.p_min_threshold.get() > self.p_max_threshold.get():
            self.p_min_threshold.set(self.p_max_threshold.get())
        self.p_min_display.config(text=f"{self.p_min_threshold.get():.2f}")

    def update_max_threshold_label(self, event=None):
        """Update the display label for voter accuracy upper threshold."""
        if self.p_max_threshold.get() < self.p_min_threshold.get():
            self.p_max_threshold.set(self.p_min_threshold.get())
        self.p_max_display.config(text=f"{self.p_max_threshold.get():.2f}")

    def validate_n_spinbox(self, value):
        """Validate the number of voters spinbox input."""
        try:
            n = int(value)
            return 2 <= n <= 10000
        except ValueError:
            return False

    def validate_inputs(self):
        """Validate user inputs before running simulation."""
        try:
            n = self.n_voters.get()
            if n < 2:
                raise ValueError("Number of voters must be at least 2.")
        except tk.TclError:
            messagebox.showerror("Input Error", "Please enter a valid integer for voters.")
            return None, None

        if not self.advanced_mode.get():
            try:
                p = self.p.get()
                if not 0 <= p <= 1:
                    raise ValueError("Voter accuracy (p) must be between 0 and 1.")
            except tk.TclError:
                messagebox.showerror("Input Error", "Please enter a valid number for voter accuracy (p).")
                return None, None
            return n, p
        else:
            try:
                sim_runs = self.simulation_runs.get()
                if sim_runs < 128:
                    raise ValueError("Simulation runs must be at least 128.")
            except tk.TclError:
                messagebox.showerror("Input Error", "Please enter a valid number for simulation runs.")
                return None, None

            try:
                p_min = self.p_min_threshold.get()
                p_max = self.p_max_threshold.get()
                if not (0 <= p_min <= 1 and 0 <= p_max <= 1):
                    raise ValueError("Voter accuracy thresholds must be between 0 and 1.")
                if p_min > p_max:
                    raise ValueError("Lower threshold cannot be greater than upper threshold.")
            except tk.TclError:
                messagebox.showerror("Input Error", "Please enter valid numbers for voter accuracy thresholds.")
                return None, None

            return (n, (self.simulation_runs.get(), self.p_min_threshold.get(), self.p_max_threshold.get()))

    def run_simulation(self):
        """Run the Condorcet simulation with validated inputs."""
        if self.is_closing:
            return

        self.run_button.config(state='disabled')

        if self.plot_canvas:
            self.plot_canvas.get_tk_widget().destroy()
            self.plot_canvas = None

        self.progress.grid()
        self.progress.config(mode='indeterminate')
        self.progress.start(10)

        thread = threading.Thread(target=self.perform_simulation, daemon=True)
        thread.start()

    def perform_simulation(self):
        """Perform the simulation computation."""
        validated = self.validate_inputs()
        if validated[0] is None:
            self.finish_simulation()
            return

        n = validated[0]
        if not self.advanced_mode.get():
            p = validated[1]
            try:
                majority_success_prob = self.calculate_condorcet_optimized(n, p)
                if self.is_closing:
                    return
                self.root.after(0, self.update_result, f"{majority_success_prob:.4f}")
                if self.is_closing:
                    return
                self.root.after(0, self.plot_probability_vs_p, n)
            except Exception as e:
                if not self.is_closing:
                    self.root.after(0, self.show_calculation_error, str(e))
        else:
            sim_runs, p_min, p_max = validated[1]
            try:
                # Check if p_min equals p_max
                if abs(p_min - p_max) < 1e-10:  # Using small epsilon for floating-point comparison
                    # Use basic mode calculation instead of simulation
                    majority_success_prob = self.calculate_condorcet_optimized(n, p_min)
                    result_text = f"{majority_success_prob:.4f} (Exact calculation at p = {p_min:.2f})"
                    if self.is_closing:
                        return
                    self.root.after(0, self.update_result, result_text)
                    if self.is_closing:
                        return
                    # Use the basic mode plot with p_min highlighted
                    self.p.set(p_min)  # Temporarily set p for plotting
                    self.root.after(0, self.plot_probability_vs_p, n)
                else:
                    # Perform normal LHS simulation
                    majority_success_prob = self.calculate_condorcet_lhs(n, sim_runs, p_min, p_max)
                    result_text = (f"{majority_success_prob:.4f} (Estimated via {sim_runs} LHS samples with "
                                f"{p_min:.2f} ≤ p_i ≤ {p_max:.2f})")
                    if self.is_closing:
                        return
                    self.root.after(0, self.update_result, result_text)
                    if self.is_closing:
                        return
                    self.root.after(0, self.plot_advanced_mode_lhs, n, sim_runs, p_min, p_max)
            except MemoryError:
                if not self.is_closing:
                    self.root.after(0, self.show_calculation_error,
                                  "Simulation requires too much memory. "
                                  "Reduce the number of voters or simulation runs.")
            except Exception as e:
                if not self.is_closing:
                    self.root.after(0, self.show_calculation_error, str(e))
        self.root.after(0, self.finish_simulation)

    def calculate_condorcet_optimized(self, n, p):
        """Calculate Condorcet probability using normal approximation for large n."""
        if p == 1.0:
            return 1.0
        if p == 0.0:
            return 0.0

        if n < 1000:
            return self.calculate_condorcet_exact(n, p)
        else:
            mean = n * p
            std = np.sqrt(n * p * (1 - p)) if p not in [0, 1] else 1e-10
            majority = n / 2
            z = (majority - mean) / std
            return 1 - special.ndtr(z)

    def calculate_condorcet_exact(self, n, p):
        """Exact calculation for small n using binomial probabilities."""
        if p == 1.0:
            return 1.0
        if p == 0.0:
            return 0.0

        majority = n // 2 + 1
        k = np.arange(majority, n + 1)
        log_probs = (
            special.gammaln(n + 1) -
            special.gammaln(k + 1) -
            special.gammaln(n - k + 1)
        ) + k * np.log(p) + (n - k) * np.log(1 - p)
        probs = np.exp(log_probs)
        prob = np.sum(probs)
        return min(1.0, max(0.0, prob))

    def calculate_condorcet_lhs(self, n, sim_runs, p_min, p_max):
        """
        Calculate majority success probability using Latin Hypercube Sampling.
        Each voter has an accuracy p_i sampled using LHS in range [p_min, p_max].
        """
        if abs(p_min - p_max) < 1e-10:  # Using small epsilon for floating-point comparison
            return self.calculate_condorcet_optimized(n, p_min)

        # Create LHS sampler for n-dimensional space
        sampler = qmc.LatinHypercube(d=n)
        
        # Generate LHS samples
        samples = sampler.random(n=sim_runs)
        
        # Scale samples to [p_min, p_max] range
        p_i = qmc.scale(samples, p_min, p_max)
        
        # Generate votes using the LHS-sampled probabilities
        votes = np.random.binomial(1, p_i)
        
        # Count successful majority decisions
        sums = np.sum(votes, axis=1)
        majority_success = np.sum(sums > (n / 2))
        
        return majority_success / sim_runs

    def plot_probability_vs_p(self, n):
        """Plot the probability that the majority is correct as a function of p."""
        plt.close('all')

        # Generate p values
        p_values = np.linspace(0, 1, 500)
        prob_values = np.array([self.calculate_condorcet_optimized(n, p) for p in p_values])

        # Create a figure
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        ax.plot(p_values, prob_values, label='Majority Correct Probability', color='blue')
        ax.axvline(self.p.get(), color='red', linestyle='--', label=f'p = {self.p.get():.2f}')
        ax.set_title(f'Condorcet Probability vs Voter Accuracy (n={n})')
        ax.set_xlabel('Voter Accuracy (p)')
        ax.set_ylabel('Probability Majority is Correct')
        ax.legend()
        ax.grid(True)

        # Embed the plot in Tkinter
        self.plot_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_advanced_mode_lhs(self, n, sim_runs, p_min, p_max):
        """Plot results for Advanced Mode using LHS data stacked horizontally."""
        plt.close('all')
        
        # Check if p_min equals p_max
        if abs(p_min - p_max) < 1e-10:
            # Use basic mode plot instead
            self.plot_probability_vs_p(n)
            return
        
        # Create LHS samples for visualization
        sampler = qmc.LatinHypercube(d=n)
        samples = sampler.random(n=sim_runs)
        p_i = qmc.scale(samples, p_min, p_max)
        votes = np.random.binomial(1, p_i)
        sums = np.sum(votes, axis=1)
    
        # Create a horizontally stacked figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=100)  # Squished horizontally
        
        # Plot Vote Distribution on the first axis
        ax1 = axes[0]
        bins = np.linspace(0, n, min(n + 1, 50))
        ax1.hist(sums, bins=bins, edgecolor='black', alpha=0.7, color='skyblue')
        ax1.axvline(n / 2, color='red', linestyle='--', label='Majority Threshold')
        ax1.set_title(f'Distribution of Correct Votes\n(n={n}, {p_min:.2f} ≤ p_i ≤ {p_max:.2f})')
        ax1.set_xlabel('Number of Correct Votes')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True)
        
        # Plot Accuracy Distribution on the second axis
        ax2 = axes[1]
        ax2.hist(p_i.flatten(), bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
        ax2.set_xlabel('Voter Accuracy')
        ax2.set_ylabel('Frequency')
        ax2.set_title('LHS Sample Distribution')
        ax2.grid(True)
    
        # Embed the plots into the Tkinter interface
        self.plot_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    

    def update_result(self, message):
        """Update the result display with the given message."""
        self.result_value.config(text=message)

    def show_calculation_error(self, message):
        """Show a calculation error message."""
        messagebox.showerror("Calculation Error", message)

    def finish_simulation(self):
        """Reset UI elements after simulation."""
        self.progress.stop()
        self.progress.grid_remove()
        self.run_button.config(state='normal')

    def on_closing(self):
        """Handle the window close event for clean termination."""
        self.is_closing = True
        plt.close('all')
        if self.plot_canvas:
            self.plot_canvas.get_tk_widget().destroy()
            self.plot_canvas = None
        self.root.destroy()


def main():
    root = tk.Tk()
    app = CondorcetSimulation(root)
    root.mainloop()


if __name__ == "__main__":
    main()