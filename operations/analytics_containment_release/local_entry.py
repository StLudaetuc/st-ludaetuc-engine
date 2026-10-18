"""Disposable loopback rehearsal factory. NEVER a production entrypoint."""
import os

def create_app():
    if any(os.environ.get(k) for k in ('DATABASE_URL','PGSERVICE','PGHOST','PGPASSWORD','ANALYTICS_SERVICE_IDENTITIES_JSON')):
        raise RuntimeError('clean local environment required')
    import psycopg2,psycopg2.pool
    def blocked(*a,**kw):raise AssertionError('DATABASE_CONNECTION_PROHIBITED')
    psycopg2.connect=blocked
    psycopg2.pool.ThreadedConnectionPool=blocked
    import engine
    engine._get_pool=blocked
    if os.environ.get('OPS_REHEARSAL_MODE')=='fallback':
        from security.analytics_release.containment_fallback import ContainmentOnly
        return ContainmentOnly(engine.app)
    return engine.app
