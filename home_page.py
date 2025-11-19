from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QGuiApplication, QColor
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QGraphicsDropShadowEffect, 
    QScrollArea, QFrame, QSizePolicy, QGridLayout
)

from PySide6.QtCore import Qt
from config import PALETTE

def drop_shadow(radius=40, dx=0, dy=12, color=QColor(109, 79, 232, 50)):
    eff = QGraphicsDropShadowEffect()
    eff.setBlurRadius(radius)
    eff.setOffset(dx, dy)
    eff.setColor(color)
    return eff

class LandingPage(QWidget):
    def __init__(self, on_cta):
        super().__init__()
        self.setStyleSheet(f"background: {PALETTE['bg']};")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet(f"""
            QScrollArea {{ background: {PALETTE['bg']}; border: none; }}
            QScrollBar:vertical {{
                background: {PALETTE['bg']}; width: 10px; border-radius: 5px;
            }}
            QScrollBar::handle:vertical {{
                background: {PALETTE['border']}; border-radius: 5px; min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{ background: {PALETTE['primary']}; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
        """)
        root.addWidget(scroll)

        container = QWidget()
        container.setStyleSheet(f"background: {PALETTE['bg']};")
        scroll.setWidget(container)

        outer = QVBoxLayout(container)
        outer.setContentsMargins(24, 40, 24, 40)
        outer.setSpacing(0)
        outer.setAlignment(Qt.AlignCenter)

        self.card = QFrame(objectName="HeroCard")
        self.card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.card.setMaximumWidth(1600)
        self.card.setGraphicsEffect(drop_shadow())

        self.card.setStyleSheet(f"""
            QFrame#HeroCard {{
                background: {PALETTE['surface']};
                border-radius: 16px;
                border: 1px solid {PALETTE['border']};
            }}
            QLabel#Heading {{ color: {PALETTE['text']}; background: transparent; }}
            QLabel#Subtitle {{ color: {PALETTE['accent']}; background: transparent; font-weight: 600; }}
            QLabel#Body {{ color: {PALETTE['muted']}; background: transparent; }}
            QLabel#FeatureTitle {{ color: {PALETTE['text']}; background: transparent; font-weight: 600; }}
            QLabel#FeatureDesc {{ color: {PALETTE['muted']}; background: transparent; }}
            QFrame#FeatureCard {{
                background: rgba(139, 92, 246, 0.05);
                border-radius: 12px;
                border: 1px solid {PALETTE['border']};
                padding: 16px;
            }}
            QPushButton#CTA {{
                color: {PALETTE['text']};
                border-radius: 10px;
                padding: 16px 32px;
                font-weight: 600;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                            stop:0 {PALETTE['primary']}, stop:1 {PALETTE['primary2']});
                border: none;
            }}
            QPushButton#CTA:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                            stop:0 {PALETTE['primary2']}, stop:1 {PALETTE['primary']});
            }}
            QPushButton#CTA:pressed {{ background: {PALETTE['glow']}; }}
        """)
        outer.addWidget(self.card, alignment=Qt.AlignCenter)

        self.card_layout = QVBoxLayout(self.card)
        self.card_layout.setSpacing(24)
        self.card_layout.setContentsMargins(32, 32, 32, 32)

        header_layout = QVBoxLayout()
        header_layout.setSpacing(8)

        self.heading = QLabel("CerebroX", objectName="Heading")
        self.heading.setWordWrap(True)
        self.heading.setAlignment(Qt.AlignCenter)
        self.heading.setFont(QFont("Segoe UI", 36, QFont.Bold))

        self.subtitle = QLabel(
            "Automated XLSX Data Cleaning & Analysis Software<br>" + 
            f"Designed by <span style='color: {PALETTE['text']};'>Eng. Judah Sleibi</span>", 
            objectName="Subtitle"
        )
        self.subtitle.setWordWrap(True)
        self.subtitle.setAlignment(Qt.AlignCenter)
        self.subtitle.setFont(QFont("Segoe UI", 16))

        header_layout.addWidget(self.heading)
        header_layout.addWidget(self.subtitle)

        self.top_divider = QFrame()
        self.top_divider.setFrameShape(QFrame.HLine)
        self.top_divider.setStyleSheet(f"background: {PALETTE['border']}; max-height: 1px;")
        header_layout.addWidget(self.top_divider)

        self.card_layout.addLayout(header_layout)

        body_html = f"""
        <p style="margin:0; line-height:1.6; color:{PALETTE['muted']}; text-align:center; font-size:15px;">
            Transform complex XLSX files into clean, validated datasets with intelligent error detection,
            automated corrections, and insightful visualizations — all via a single configuration file.
        </p>
        """
        self.body = QLabel(body_html, objectName="Body")
        self.body.setWordWrap(True)
        self.body.setTextFormat(Qt.RichText)
        self.body.setAlignment(Qt.AlignCenter)
        self.body.setFont(QFont("Segoe UI", 14))
        self.card_layout.addWidget(self.body)

        self.features_container = QWidget()
        self.features_container.setStyleSheet("background: transparent;")
        self.features_layout = QGridLayout(self.features_container)
        self.features_layout.setSpacing(20)
        self.features_layout.setContentsMargins(0, 8, 0, 8)
        
        self.features_layout.setColumnStretch(0, 1)
        self.features_layout.setColumnStretch(1, 1)

        self.features = [
            ("⚡", "Intelligent Cleaning", "Automatically detect and fix data inconsistencies, errors, and formatting issues."),
            ("📊", "Rich Visualizations", "Generate publication-ready charts and insights from your cleaned data."),
            ("⚙️", "Config-Driven", "Control the entire pipeline with a single, elegant YAML configuration."),
            ("🔍", "Smart Validation", "Built-in validation rules ensure data quality and integrity throughout."),
        ]

        self.feature_cards = []
        for icon, title, desc in self.features:
            c = self._create_feature_card(icon, title, desc)
            self.feature_cards.append(c)

        self.card_layout.addWidget(self.features_container)

        cta_layout = QVBoxLayout()
        cta_layout.setSpacing(16)
        cta_layout.setAlignment(Qt.AlignCenter)

        self.cta = QPushButton("Get Started", objectName="CTA")
        self.cta.setCursor(Qt.PointingHandCursor)
        self.cta.clicked.connect(on_cta)
        self.cta.setMinimumHeight(52)
        self.cta.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.cta.setFont(QFont("Segoe UI", 15, QFont.DemiBold))

        cta_layout.addWidget(self.cta)
        self.card_layout.addLayout(cta_layout)

        divider_bottom = QFrame()
        divider_bottom.setFrameShape(QFrame.HLine)
        divider_bottom.setStyleSheet(f"background: {PALETTE['border']}; max-height: 1px;")
        self.card_layout.addWidget(divider_bottom)

        credit = QLabel(
            f'<span style="color:{PALETTE["muted"]}; font-size:16px;">Developed by </span>'
            f'<b style="color:{PALETTE["text"]}; font-size:16px;">Eng. Judah Sleibi</b>'
            f'<span style="color:{PALETTE["muted"]}; font-size:16px;"> • </span>'
            f'<a style="color:{PALETTE["accent"]}; text-decoration:none; font-weight:600; font-size:16px;" '
            f'href="https://www.linkedin.com/in/judah-sleibi-b8578b321/">LinkedIn</a>'
        )
        credit.setOpenExternalLinks(True)
        credit.setTextFormat(Qt.RichText)
        credit.setAlignment(Qt.AlignCenter)
        credit.setStyleSheet("background: transparent;")
        self.card_layout.addWidget(credit)

        self._reflow_feature_grid(self.card.width())
        self.update_typography(self.logical_width())

    def _create_feature_card(self, icon, title, description):
        card = QFrame(objectName="FeatureCard")
        card.setMinimumWidth(120)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        layout = QVBoxLayout(card)
        layout.setSpacing(10)
        layout.setContentsMargins(18, 18, 18, 18)

        icon_label = QLabel(icon)
        icon_label.setFont(QFont("Segoe UI", 26))
        icon_label.setStyleSheet("background: transparent;")
        icon_label.setAlignment(Qt.AlignLeft)

        title_label = QLabel(title, objectName="FeatureTitle")
        title_label.setFont(QFont("Segoe UI", 14, QFont.DemiBold))
        title_label.setWordWrap(True)
        title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        desc_label = QLabel(description, objectName="FeatureDesc")
        desc_label.setFont(QFont("Segoe UI", 12))
        desc_label.setWordWrap(True)
        desc_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        layout.addWidget(icon_label)
        layout.addWidget(title_label)
        layout.addWidget(desc_label)
        layout.addStretch()

        return card

    def _clear_feature_grid(self):
        while self.features_layout.count():
            item = self.features_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

    def _reflow_feature_grid(self, width: int):
        self._clear_feature_grid()

        two_cols = width >= 1300
        if two_cols:
            col_min = max(600, int((width - 40) / 2))
            self.features_layout.setColumnMinimumWidth(0, col_min)
            self.features_layout.setColumnMinimumWidth(1, col_min)
            for idx, card in enumerate(self.feature_cards):
                r, c = divmod(idx, 2)
                self.features_layout.addWidget(card, r, c)
        else:
            self.features_layout.setColumnMinimumWidth(0, 600)
            for r, card in enumerate(self.feature_cards):
                self.features_layout.addWidget(card, r, 0)

        self.features_layout.setColumnStretch(0, 1)
        self.features_layout.setColumnStretch(1, 1)

    def logical_width(self):
        screen = QGuiApplication.primaryScreen()
        return int(screen.availableGeometry().width() / max(1, screen.devicePixelRatio()))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        w = self.width()
        self.update_typography(w)
        self._reflow_feature_grid(self.card.width())

    def update_typography(self, w: int):
        def clamp(min_px, ideal_ratio, max_px):
            return max(min_px, min(int(w * ideal_ratio), max_px))

        self.card.setMaximumWidth(clamp(1000, 0.96, 1600))

        if w < 768:
            self.heading.setFont(QFont("Segoe UI", 28, QFont.Bold))
            self.subtitle.setFont(QFont("Segoe UI", 14))
        else:
            self.heading.setFont(QFont("Segoe UI", 36, QFont.Bold))
            self.subtitle.setFont(QFont("Segoe UI", 16))
