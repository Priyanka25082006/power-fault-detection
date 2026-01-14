"""
email_alerts.py - Email notification system for fault alerts
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import os
from datetime import datetime
from dataclasses import dataclass
from typing import List
import ssl


@dataclass
class FaultData:
    """Data class for fault information"""
    timestamp: str
    current: float
    voltage: float
    fault_type: str
    confidence: float
    location: str
    severity: str


class EmailNotifier:
    """Email notification system for power faults"""

    def __init__(self, config_file='config/email_config.json'):
        self.config = self.load_config(config_file)
        self.enabled = self.config.get('enabled', False)
        self.smtp_server = self.config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = self.config.get('smtp_port', 587)
        self.sender_email = self.config.get('sender_email', '')
        self.sender_password = self.config.get('sender_password', '')
        self.recipients = self.config.get('recipients', [])

        print(f"📧 Email Notifier initialized (Enabled: {self.enabled})")

    def load_config(self, config_file):
        """Load email configuration"""
        default_config = {
            "enabled": False,  # Disabled by default for safety
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "sender_email": "sender@gmail.com",
            "sender_password": "******",
            "recipients": ["example1@gmail.com"],
            "send_test_on_startup": False,
            "email_template": "default"
        }

        try:
            os.makedirs('config', exist_ok=True)
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    return {**default_config, **config}
            else:
                # Create config file with empty credentials
                with open(config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
                return default_config
        except:
            return default_config

    def create_email_body(self, fault_data: FaultData) -> str:
        """Create HTML email body for fault alert"""

        # Determine color based on severity
        if fault_data.severity == "CRITICAL":
            severity_color = "#DC2626"  # Red
            severity_icon = "🔴"
        elif fault_data.severity == "WARNING":
            severity_color = "#F59E0B"  # Orange
            severity_icon = "🟡"
        else:
            severity_color = "#3B82F6"  # Blue
            severity_icon = "🔵"

        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: {severity_color}; color: white; padding: 20px; text-align: center; border-radius: 5px 5px 0 0; }}
                .content {{ background-color: #f9f9f9; padding: 20px; border-radius: 0 0 5px 5px; }}
                .alert-box {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 15px 0; }}
                .metrics {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin: 20px 0; }}
                .metric {{ background-color: white; padding: 10px; border-radius: 5px; border-left: 4px solid {severity_color}; }}
                .footer {{ margin-top: 20px; padding-top: 20px; border-top: 1px solid #eee; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{severity_icon} POWER SYSTEM FAULT ALERT</h1>
                    <h2>{fault_data.severity} - {fault_data.fault_type}</h2>
                </div>

                <div class="content">
                    <div class="alert-box">
                        <h3>🚨 IMMEDIATE ATTENTION REQUIRED</h3>
                        <p>A fault has been detected in the power grid system. Please review the details below and take appropriate action.</p>
                    </div>

                    <h3>📋 Fault Details</h3>
                    <div class="metrics">
                        <div class="metric">
                            <strong>Timestamp</strong><br>
                            {fault_data.timestamp}
                        </div>
                        <div class="metric">
                            <strong>Fault Type</strong><br>
                            {fault_data.fault_type}
                        </div>
                        <div class="metric">
                            <strong>Current</strong><br>
                            {fault_data.current:.1f} A
                        </div>
                        <div class="metric">
                            <strong>Voltage</strong><br>
                            {fault_data.voltage:.1f} V
                        </div>
                        <div class="metric">
                            <strong>Confidence</strong><br>
                            {fault_data.confidence:.1%}
                        </div>
                        <div class="metric">
                            <strong>Location</strong><br>
                            {fault_data.location}
                        </div>
                    </div>

                    <h3>🚨 Recommended Actions</h3>
                    <ol>
                        <li>Dispatch maintenance team to the affected location</li>
                        <li>Isolate the faulty section from the grid</li>
                        <li>Initiate backup power routing if available</li>
                        <li>Notify grid control center immediately</li>
                        <li>Monitor for any secondary faults</li>
                    </ol>

                    <div class="footer">
                        <p>This is an automated alert from the Power System Fault Detection System.</p>
                        <p>⚠️ DO NOT REPLY TO THIS EMAIL</p>
                        <p>Report ID: FLT-{datetime.now().strftime('%Y%m%d%H%M%S')}</p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        return html_body

    def send_alert(self, fault_data: FaultData) -> dict:
        """Send email alert for fault"""

        if not self.enabled:
            return {"success": False, "message": "Email alerts are disabled"}

        if not self.sender_email or not self.sender_password:
            return {"success": False, "message": "Email credentials not configured"}

        if not self.recipients:
            return {"success": False, "message": "No recipients configured"}

        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"⚡ {fault_data.severity}: {fault_data.fault_type} - Power System Fault"
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(self.recipients)

            # Create plain text version
            text = f"""
            POWER SYSTEM FAULT ALERT
            ========================

            Severity: {fault_data.severity}
            Fault Type: {fault_data.fault_type}
            Time: {fault_data.timestamp}

            Parameters:
            - Current: {fault_data.current:.1f} A
            - Voltage: {fault_data.voltage:.1f} V
            - Confidence: {fault_data.confidence:.1%}
            - Location: {fault_data.location}

            Recommended Actions:
            1. Dispatch maintenance team
            2. Isolate affected section
            3. Initiate backup power
            4. Notify control center

            This is an automated alert. DO NOT REPLY.
            """

            # Create HTML version
            html = self.create_email_body(fault_data)

            # Attach both versions
            part1 = MIMEText(text, 'plain')
            part2 = MIMEText(html, 'html')

            msg.attach(part1)
            msg.attach(part2)

            # Send email
            context = ssl.create_default_context()

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)

            print(f"📧 Email alert sent to {len(self.recipients)} recipients")
            return {"success": True, "message": "Email sent successfully"}

        except Exception as e:
            error_msg = f"Failed to send email: {str(e)}"
            print(f"❌ {error_msg}")
            return {"success": False, "message": error_msg}

    def test_connection(self) -> dict:
        """Test email connection and credentials"""
        if not self.enabled:
            return {"success": False, "message": "Email alerts are disabled"}

        if not self.sender_email or not self.sender_password:
            return {"success": False, "message": "Email credentials not configured"}

        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)

            return {"success": True, "message": "Connection test successful"}

        except Exception as e:
            return {"success": False, "message": f"Connection failed: {str(e)}"}


# Quick test
if __name__ == "__main__":
    # Test the email system
    notifier = EmailNotifier()

    if notifier.enabled:
        print("Testing email system...")

        # Test connection
        result = notifier.test_connection()
        print(f"Connection test: {result['message']}")

        if result['success']:
            # Send test email
            test_fault = FaultData(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                current=350.0,
                voltage=80.0,
                fault_type="TEST_FAULT",
                confidence=0.95,
                location="Test Location - Grid Section 5",
                severity="TEST"
            )

            send_result = notifier.send_alert(test_fault)
            print(f"Send test: {send_result['message']}")
    else:
        print("Email alerts are disabled. Enable them in config/email_config.json")