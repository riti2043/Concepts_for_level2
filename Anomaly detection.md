---


---

<h2 id="anomaly-detection">Anomaly Detection</h2>
<h3 id="what-is-anomaly-detection">What is Anomaly detection?</h3>
<p>Anomaly Detection is the process of identifying rare events or outliers that deviate significantly from the “normal” in a dataset.These anomalies can signal potential issues or intresting patterns in the data.<br>
What is considered “normal” depends heavily on context.</p>
<h3 id="types-of-anomalies-">Types of Anomalies :</h3>
<p>This classification distinguishes anomalies by <em>how</em> they stand out relative to the rest of the data.<br>
• <strong>Point Anomalies:</strong>  These are individual data instances that deviate significantly from the norm on their own.<br>
<em>Example</em>: A massive credit card withdrawal that is far higher than the user’s typical spending limit.<br>
• <strong>Contextual Anomalies:</strong>  These are instances considered anomalous only within a specific context or condition; in a different context, they might be considered normal.<br>
<em>Example:</em> A low temperature (e.g., 0°C) is normal in winter (January) but would be a contextual anomaly if recorded in summer (June or July).<br>
• <strong>Collective Anomalies:</strong>  These involve a group of data points that exhibit anomalous behaviour when considered together, even if the individual points might appear normal in isolation.<br>
<em>Example:</em> In a computer lab, if one computer shuts down, it is normal. However, if <em>all</em> computers shut down simultaneously, it indicates a collective anomaly, such as a power failure or cyberattack.</p>
<h3 id="real-world-applications-">Real-world applications :</h3>
<p>Detecting these anomalies is crucial because “bad data” or undetected irregularities compromise statistical tests, dashboards, and machine learning models.</p>
<ul>
<li><strong>Credit Card Fraud:</strong> Flagging transactions that occur in unusual locations or sudden spikes in usage.</li>
<li><strong>Money Laundering &amp; Identity Theft:</strong> Identifying unusual transaction patterns that signal illicit activity.</li>
<li><strong>Insurance Claims:</strong> Flaging suspicious claim data for further investigation regarding fraud.</li>
<li><strong>Intrusion Detection Systems:</strong> Spotting abrupt increases in data transfer or the use of unknown protocols which may signal a breach or malware.</li>
<li><strong>IP Address Monitoring:</strong> Alerting administrators when an unknown IP address appears on a secure network.</li>
<li><strong>Patient Vital Signs:</strong> Monitoring for dangerous combinations, such as a sudden increase in heart rate accompanied by a decrease in blood pressure.</li>
<li><strong>Epidemiology:</strong> Identifying unusual patterns in healthcare data to detect disease outbreaks early.</li>
<li><strong>Quality Control:</strong> Automating the detection of defective products in manufacturing.</li>
<li><strong>Energy Grid Monitoring:</strong> Identifying unusual surges in power or grid instability.</li>
<li><strong>E-commerce:</strong> Analyzing user navigation to detect bot activity.</li>
</ul>
<h3 id="anomaly-detection-algorithms-"><strong>Anomaly Detection Algorithms :</strong></h3>
<p><strong>A. Statistical / Univariate Methods</strong><br>
• Z-Score: Measures how many standard deviations a point is from the mean. Usually, a score &gt; 3 is an outlier.</p>
<p>• Interquartile Range (IQR): Uses quartiles (Q1 and Q3). Outliers are defined as points outside [Q1−1.5×IQR,Q3+1.5×IQR].</p>
<p>• Modified Z-Scores: Uses Median Absolute Deviation (MAD) instead of mean/standard deviation, making it more robust against outliers.</p>
<p><strong>B. Machine Learning / Multivariate Methods</strong></p>
<p>• Local Outlier Factor (LOF): Measures the local density of a point compared to its neighbors. Points with significantly lower density than their neighbors are anomalies.</p>
<p>• Isolation Forest: Uses an ensemble of trees to isolate points. Anomalies are isolated quickly (requiring fewer splits) because they are rare and different.</p>
<p>• One-Class SVM: Learns a decision boundary that encompasses the majority (normal) data. Points falling outside this boundary are anomalies.</p>
<p>• Elliptic Envelope: Assumes data is normally distributed. It fits an ellipse around the central data points; those outside are outliers.</p>

