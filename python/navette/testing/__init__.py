"""Tools to test the runtime and robot adapters without hardware.

  fakes.py        a fake robot (FakeWorld, FakeCamera, FakeImu, FakeBase) and test doubles of the core classes
                  the runtime gets injected (FakeNavigator, FakeUpFilter)
  conformance.py  checks of an adapter against navette.interfaces (units, signs, clocks, dead-man, status)
"""
