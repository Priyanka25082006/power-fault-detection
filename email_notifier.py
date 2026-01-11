import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
from datetime import datetime
from typing import Dict, List
import logging


class EmailNotifier:
    """Send email alerts for power line faults"""

    def __init__(self, config_file='config/email_config.json'):
        self.config = self.load_config(config_file)
        self.setup_logging()

    def load_config(self, config_file):
        """Load email configuration"""
        default_config = {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "sender_email": "priyanece08@gmail.com",
            "sender_password": "kppwgbvmxxpcrvxz",  # Use App Password, not regular password
            "recipients": ["recipient1@example.com", "recipient2@example.com"],
            "alert_cooldown_minutes": 5
        }

        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                return {**default_config, **config}
        except FileNotFoundError:
            # Create config file with instructions
            os.makedirs('config', exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            print(f"Created config file: {config_file}. Please update with your email details.")
            return default_config

    def setup_logging(self):
        """Setup logging for email notifications"""
        logging.basicConfig(
            filename='logs/email_notifications.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def create_fault_email(self, fault_data: Dict) -> MIMEMultipart:
        """Create email content for fault alert"""

        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"🚨 POWER LINE FAULT DETECTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        msg['From'] = self.config['sender_email']
        msg['To'] = ", ".join(self.config['recipients'])

        # Plain text version
        text = f"""
        CRITICAL POWER LINE FAULT DETECTED!

        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        Current Readings:
        - Current: {fault_data.get('current', 'N/A')} A
        - Voltage: {fault_data.get('voltage', 'N/A')} V
        - Power: {fault_data.get('power', 'N/A')} W
        - Frequency: {fault_data.get('frequency', 'N/A')} Hz

        Fault Details:
        {fault_data.get('fault_description', 'Unknown fault type')}

        Confidence: {fault_data.get('confidence', 0):.2%}

        Location: {fault_data.get('location', 'Unknown')}

        Action Required: Immediate attention needed!

        ---
        This is an automated alert from AI Power Line Fault Detection System.
        """

        # HTML version
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .alert {{ background-color: #ffe6e6; border-left: 5px solid #ff3333; padding: 15px; }}
                .reading {{ background-color: #f0f0f0; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                .action {{ background-color: #fff3cd; padding: 15px; border: 1px solid #ffc107; border-radius: 5px; }}
                .footer {{ margin-top: 20px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="alert">
                <h2 style="color: #ff3333;">🚨 CRITICAL POWER LINE FAULT DETECTED</h2>
            </div>

            <h3>📊 Current Readings</h3>
            <div class="reading">
                <strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                <strong>Current:</strong> {fault_data.get('current', 'N/A')} A<br>
                <strong>Voltage:</strong> {fault_data.get('voltage', 'N/A')} V<br>
                <strong>Power:</strong> {fault_data.get('power', 'N/A')} W<br>
                <strong>Frequency:</strong> {fault_data.get('frequency', 'N/A')} Hz
            </div>

            <h3>⚠️ Fault Details</h3>
            <div class="reading">
                <strong>Type:</strong> {fault_data.get('fault_type', 'Unknown')}<br>
                <strong>Description:</strong> {fault_data.get('fault_description', 'Unknown')}<br>
                <strong>Confidence:</strong> {fault_data.get('confidence', 0):.2%}<br>
                <strong>Severity:</strong> {fault_data.get('severity', 'Unknown')}
            </div>

            <h3>📍 Location</h3>
            <div class="reading">
                <strong>Coordinates:</strong> {fault_data.get('coordinates', 'N/A')}<br>
                <strong>Nearest Substation:</strong> {fault_data.get('substation', 'Unknown')}<br>
                <strong>Distance:</strong> {fault_data.get('distance', 'N/A')} km from substation
            </div>

            <div class="action">
                <h3>🚀 Action Required</h3>
                <p>Immediate attention needed! Please dispatch maintenance team to the location.</p>
            </div>

            <div class="footer">
                <p>This is an automated alert from AI Power Line Fault Detection System.</p>
                <p>System ID: {fault_data.get('system_id', 'FDS-001')}</p>
            </div>
        </body>
        </html>
        """

        # Attach both versions
        msg.attach(MIMEText(text, 'plain'))
        msg.attach(MIMEText(html, 'html'))

        return msg

    def send_email(self, fault_data: Dict) -> bool:
        """Send email alert"""

        # Check cooldown period
        last_sent = self.get_last_sent_time()
        cooldown = self.config['alert_cooldown_minutes'] * 60  # Convert to seconds

        if datetime.now().timestamp() - last_sent < cooldown:
            logging.info(f"Email cooldown active. Skipping email.")
            return False

        try:
            # Create email
            msg = self.create_fault_email(fault_data)

            # Connect to SMTP server
            with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
                server.starttls()
                server.login(self.config['sender_email'], self.config['sender_password'])

                # Send email to all recipients
                for recipient in self.config['recipients']:
                    server.sendmail(
                        self.config['sender_email'],
                        recipient,
                        msg.as_string()
                    )
                    logging.info(f"Email sent to {recipient}")

            # Update last sent time
            self.update_last_sent_time()

            return True

        except Exception as e:
            logging.error(f"Failed to send email: {str(e)}")
            return False

    def get_last_sent_time(self) -> float:
        """Get timestamp of last email sent"""
        try:
            with open('logs/last_email_sent.txt', 'r') as f:
                return float(f.read().strip())
        except:
            return 0

    def update_last_sent_time(self):
        """Update timestamp of last email sent"""
        with open('logs/last_email_sent.txt', 'w') as f:
            f.write(str(datetime.now().timestamp()))


# Integration with dashboard
def integrate_email_with_dashboard():
    """Add email notification to dashboard"""

    notifier = EmailNotifier()

    # In your fault detection loop
    def check_and_notify(fault_data):
        if fault_data['has_fault'] and fault_data['confidence'] > 0.8:
            # Prepare email data
            email_data = {
                'current': fault_data['reading']['current'],
                'voltage': fault_data['reading']['voltage'],
                'power': fault_data['reading']['power'],
                'frequency': fault_data['reading']['frequency'],
                'fault_type': fault_data['faults'][0]['type'] if fault_data['faults'] else 'Unknown',
                'fault_description': '; '.join([f['type'] for f in fault_data['faults']]),
                'confidence': fault_data.get('confidence', 0.95),
                'severity': 'HIGH' if fault_data.get('confidence', 0) > 0.9 else 'MEDIUM',
                'location': 'Grid Section A-12',  # From location tracking
                'coordinates': '40.7128° N, 74.0060° W',
                'substation': 'Substation NYC-05',
                'distance': '3.2',
                'system_id': 'FDS-NYC-001'
            }

            # Send email
            success = notifier.send_email(email_data)

            if success:
                st.success("📧 Email alert sent!")
            else:
                st.warning("⚠️ Email notification failed")

    return check_and_notify