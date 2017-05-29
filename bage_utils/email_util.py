import base64
import os
import re
import smtplib
import traceback
from email import utils
from email.header import Header
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class EmailUtil(object):
    def __init__(self, from_user, from_passwd, smtp_host, smtp_port=25, print_debug=False):
        _username_n_userid = re.split('<|>', from_user)
        if len(_username_n_userid) != 3 or len(_username_n_userid[0]) == 0 or len(_username_n_userid[1]) == 0:
            raise Exception(
                'from_user must be name and email. e.g. 홍길동<kildong.hong@gmail.com> ')
        self.from_user = from_user
        self.from_username = _username_n_userid[0]
        self.from_userid = _username_n_userid[1]
        self.from_passwd = from_passwd
        self.host = smtp_host
        self.port = smtp_port
        self.print_debug = print_debug

    def send(self, subject, contents, to_user, to_cc_users=None, to_bcc_users=None, attach=None):
        """
        send an email with attach file.
        :param subject: email title
        :param contents: email content
        :param to_user: to user email address
        :param to_bcc_users: 
        :param to_cc_users: 
        :param attach: attach file path
        """
        if not to_bcc_users:
            to_bcc_users = []
        if not to_cc_users:
            to_cc_users = []
        try:
            print(to_user, type(to_user))
            if not isinstance(to_user, list):
                to_user = [to_user]
            if not isinstance(to_cc_users, list):
                to_cc_users = [to_cc_users]
            if not isinstance(to_bcc_users, list):
                to_bcc_users = [to_bcc_users]
            to_all_users = to_user + to_cc_users + to_bcc_users

            msg = MIMEMultipart("alternative")
            msg["From"] = self.from_user
            msg["To"] = ", ".join(to_user)
            msg["Cc"] = ", ".join(to_cc_users)
            msg["Subject"] = Header(s=subject, charset="utf-8")
            msg["Date"] = utils.formatdate(localtime=1)
            msg.attach(MIMEText(contents, "html", _charset="utf-8"))

            if attach:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(open(attach, "rb").read())
                base64.b64encode(part)
                part.add_header(
                    "Content-Disposition", "attachment; filename=\"%s\"" % os.path.basename(attach))
                msg.attach(part)

            smtp = smtplib.SMTP(self.host, self.port)
            smtp.set_debuglevel(0)
            if self.print_debug:
                smtp.set_debuglevel(1)
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()

            smtp.login(self.from_userid, self.from_passwd)
            smtp.sendmail(self.from_user, to_all_users, msg.as_string())
            smtp.close()
            return True
        except:
            traceback.print_exc()
            return False


if __name__ == '__main__':
    email = EmailUtil(from_user='홍길동<kildong.honggmail.com>', from_passwd='passwd', smtp_host='smtp.gmail.com', smtp_port=587,
                      print_debug=True)
    email.send(subject=u'제목...', contents=u'본문...', to_user='홍길동<kdhong@home.co.kr>')
