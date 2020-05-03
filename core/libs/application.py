import web
import traceback

from core.libs import log


class AmengApplication(web.application):
    def handle_with_processors(self):
        def process(processors):
            try:
                if processors:
                    p, processors = processors[0], processors[1:]
                    return p(lambda: process(processors))
                else:
                    return self.handle()
            except (web.HTTPError, KeyboardInterrupt, SystemExit):
                raise
            except:
                log.error(traceback.format_exc())
                raise self.internalerror()

        return process(self.processors)
