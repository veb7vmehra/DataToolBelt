"""Route declaration."""

from flask import Flask
from flask import render_template

app = Flask(__name__)


@app.route('/')
def home():
    """Landing page."""
    return render_template('home.html',
                           title="DataToolBelt",
                           description="Your one stop destination for all your data needs!")
