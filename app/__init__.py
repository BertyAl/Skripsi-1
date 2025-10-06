import os
from flask import Flask, render_template
from dotenv import load_dotenv

def create_app():
    load_dotenv()
    app = Flask(__name__, template_folder='templates')
    app.config.from_object('config.Config')

    # Register blueprints
    from app.main.routes import bp as main_bp

    app.register_blueprint(main_bp)

    # Error handlers
    @app.errorhandler(404)
    def not_found(e):
        return render_template('errors/404.html', title='Not Found'), 404

    @app.errorhandler(500)
    def server_error(e):
        return render_template('errors/500.html', title='Server Error'), 500

    # Jinja helpers
    @app.context_processor
    def inject_globals():
        import datetime as _dt
        return dict(now=_dt.datetime.now)

    return app
