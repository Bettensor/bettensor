import nox

@nox.session(python=["3.8", "3.9", "3.10"])
def tests(session):
    session.install("pytest")
    session.install(".")
    session.run("pytest", "tests")