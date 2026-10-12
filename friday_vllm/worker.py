"""在后台线程跑耗时的事（SSH、网络），结果回到界面线程。"""
import traceback

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class _Signals(QObject):
    done = Signal(object)
    failed = Signal(object)


class _Task(QRunnable):
    def __init__(self, function, args, kwargs):
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.signals = _Signals()

    def run(self):
        try:
            result = self.function(*self.args, **self.kwargs)
        except Exception as error:  # noqa: BLE001
            error.trace = traceback.format_exc()
            self.signals.failed.emit(error)
            return
        self.signals.done.emit(result)


_pool = QThreadPool()
_pool.setMaxThreadCount(8)
_alive = set()


def background(function, *args, done=None, failed=None, **kwargs):
    """function(*args) 在线程池里跑；done(结果) / failed(异常) 在界面线程回调"""
    task = _Task(function, args, kwargs)
    _alive.add(task.signals)
    if done:
        task.signals.done.connect(done)
    if failed:
        task.signals.failed.connect(failed)
    task.signals.done.connect(lambda _: _alive.discard(task.signals))
    task.signals.failed.connect(lambda _: _alive.discard(task.signals))
    _pool.start(task)
