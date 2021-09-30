FROM debian:stable-slim

RUN mkdir /simulator
COPY . /simulator
RUN chmod +x /simulator/linux.x86_64

CMD /simulator/linux.x86_64

