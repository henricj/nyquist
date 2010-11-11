/*
 *  Copyright (C) 2004 Steve Harris
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  $Id: server.c,v 1.3 2007/02/28 03:17:12 rbd Exp $
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <float.h>
#include <sys/types.h>

// this is probably wrong to define this here, but
// if you include xlisp.h and sound.h, you get all
// kinds of conflicts with windows.h
#include "switches.h"
#ifdef USE_PRINTF
#define nyquist_printf printf
#endif

#ifdef WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#define EADDRINUSE WSAEADDRINUSE
#define snprintf _snprintf
#else
#include <unistd.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/poll.h>
#include <sys/un.h>
#include <arpa/inet.h>
#endif

#ifdef WIN32
#define geterror() WSAGetLastError()
#else
#define geterror() errno
#endif

#include "lo_types_internal.h"
#include "lo/lo.h"
#include "lo/lo_throw.h"

#define LO_HOST_SIZE 1024

typedef struct {
    lo_timetag ts;
    size_t len;
    void *data;
    void *next;
} queued_msg_list;

struct lo_cs lo_client_sockets = {-1, -1};

static int lo_can_coerce_spec(const char *a, const char *b);
static int lo_can_coerce(char a, char b);
static void dispatch_method(lo_server s, const char *path, char *types,
			    void *data);
static int dispatch_queued(lo_server s);
static void queue_data(lo_server s, lo_timetag ts, void *data, size_t len);


#ifdef WIN32
#ifndef gai_strerror
// Copied from the Win32 SDK 

// WARNING: The gai_strerror inline functions below use static buffers,
// and hence are not thread-safe.  We'll use buffers long enough to hold
// 1k characters.  Any system error messages longer than this will be
// returned as empty strings.  However 1k should work for the error codes
// used by getaddrinfo().
#define GAI_STRERROR_BUFFER_SIZE 1024

char *WSAAPI gai_strerrorA(int ecode)
{
    DWORD dwMsgLen;
    static char buff[GAI_STRERROR_BUFFER_SIZE + 1];

    dwMsgLen = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM
                             |FORMAT_MESSAGE_IGNORE_INSERTS
                             |FORMAT_MESSAGE_MAX_WIDTH_MASK,
                              NULL,
                              ecode,
                              MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                              (LPSTR)buff,
                              GAI_STRERROR_BUFFER_SIZE,
                              NULL);
    return buff;
}
#endif

static int stateWSock = -1;

int initWSock()
{
    WORD reqversion;
    WSADATA wsaData;
    if(stateWSock >= 0) return stateWSock;
    /* TODO - which version of Winsock do we actually need? */

    reqversion = MAKEWORD( 2, 2 );
    if(WSAStartup(reqversion,&wsaData) != 0) {
        /* Couldn't initialize Winsock */
        stateWSock = 0;
    }
    else if ( LOBYTE( wsaData.wVersion ) != LOBYTE(reqversion) ||
            HIBYTE( wsaData.wVersion ) != HIBYTE(reqversion) ) {
        /* wrong version */
        WSACleanup();
        stateWSock = 0; 
    }
    else
        stateWSock = 1;

    return stateWSock;
}

// this is used here only to get a "random" number
static int time(void *p)
{
    LARGE_INTEGER tim;
    QueryPerformanceCounter(&tim);
    return (int) (tim.LowPart);
}

#define close(s) closesocket(s)
#endif

lo_server lo_server_new(const char *port, lo_err_handler err_h)
{
#ifndef WIN32
    if (port && *port == '/') {
	return lo_server_new_with_proto(port, LO_UNIX, err_h);
    } 
    else 
#endif
    {
	return lo_server_new_with_proto(port, LO_UDP, err_h);
    }
}

lo_server lo_server_new_with_proto(const char *port, int proto,
				   lo_err_handler err_h)
{
    lo_server s;
    struct addrinfo *ai = NULL, *it, *used;
    struct addrinfo hints;
    int ret = -1;
    int tries = 0;
    char pnum[16];
    const char *service;
    char hostname[LO_HOST_SIZE];

#ifdef WIN32
    if(!initWSock()) return NULL;
#endif
    
    s = calloc(1, sizeof(struct _lo_server));

    s->err_h = err_h;
    s->first = NULL;
    s->ai = NULL;
    s->hostname = NULL;
    s->protocol = proto;
    s->port = 0;
    s->path = NULL;
    s->queued = NULL;
    s->socket = -1;

    memset(&hints, 0, sizeof(hints));

    if (proto == LO_UDP) {
	hints.ai_socktype = SOCK_DGRAM;
    } else if (proto == LO_TCP) {
	hints.ai_socktype = SOCK_STREAM;
    } 
#ifndef WIN32
    else if (proto == LO_UNIX) {

	struct sockaddr_un sa;

	s->socket = socket(PF_UNIX, SOCK_DGRAM, 0);
	if (s->socket == -1) {
        int err = geterror();
	    used = NULL;
	    lo_throw(s, err, strerror(err), "socket()");
	    lo_server_free(s);

	    return NULL;
	}

	sa.sun_family = AF_UNIX;
	strncpy(sa.sun_path, port, 107);

	if ((ret = bind(s->socket, (struct sockaddr *)&sa, sizeof(sa))) < 0) {
        int err = geterror();      
	    lo_throw(s, err, strerror(err), "bind()");

	    lo_server_free(s);
	    return NULL;
	}

	s->path = strdup(port);

	return s;
    } 
#endif
    else {
	lo_throw(s, LO_UNKNOWNPROTO, "Unknown protocol", NULL);
	lo_server_free(s);

	return NULL;
    }

#ifdef ENABLE_IPV6
    hints.ai_family = PF_UNSPEC;
#else
    hints.ai_family = PF_INET;
#endif
    hints.ai_flags = AI_PASSIVE;

    if (!port) {
	service = pnum;
    } else {
	service = port;
    }
    do {
	if (!port) {
	    /* not a good way to get random numbers, but its not critical */
	    snprintf(pnum, 15, "%ld", 10000 + ((unsigned int)rand() +
		     time(NULL)) % 10000);
	}

	if (ai) {
	    freeaddrinfo(ai);
	}

	if ((ret = getaddrinfo(NULL, service, &hints, &ai))) {
	    lo_throw(s, ret, gai_strerror(ret), NULL);
	    freeaddrinfo(ai);

	    return NULL;
	}
#ifdef WIN32
        nyquist_printf("getenv(\"SystemRoot\")=%s\n", 
                       getenv("SystemRoot"));
#endif
	used = NULL;
	s->ai = ai;
	s->socket = -1;
	s->port = 0;

	for (it = ai; it && s->socket == -1; it = it->ai_next) {
	    used = it;
	    s->socket = socket(it->ai_family, hints.ai_socktype, 0);
	}
	if (s->socket == -1) {
        int err = geterror();
	    used = NULL;
	    lo_throw(s, err, strerror(err), "socket()");

	    lo_server_free(s);
	    return NULL;
	}

	if ((ret = bind(s->socket, used->ai_addr, used->ai_addrlen)) < 0) {
        int err = geterror();
	    if (err == EINVAL || err == EADDRINUSE) {
		used = NULL;

		continue;
	    }
	    lo_throw(s, err, strerror(err), "bind()");

	    lo_server_free(s);

	    return NULL;
	}
    } while (!used && tries++ < 16);

    if (proto == LO_TCP) {
	listen(s->socket, 8);
    }

    if (!used) {
	lo_throw(s, LO_NOPORT, "cannot find free port", NULL);

	lo_server_free(s);
	return NULL;
    }

    if (proto == LO_UDP) {
	lo_client_sockets.udp = s->socket;
    } else if (proto == LO_TCP) {
        lo_client_sockets.tcp = s->socket;
    }

    /* Try it the IPV6 friendly way first */
    hostname[0] = '\0';
    for (it = ai; it; it = it->ai_next) {
	if (getnameinfo(it->ai_addr, it->ai_addrlen, hostname,
			sizeof(hostname), NULL, 0, NI_NAMEREQD) == 0) {
	    break;
	}
    }

    /* check to make sure getnameinfo() didn't just set the hostname to "::".
       Needed on Darwin. */
    if (hostname[0] == ':') {
	hostname[0] = '\0';
    }

    /* Fallback to the oldschool (i.e. more reliable) way */
    if (!hostname[0]) {
	struct hostent *he;

	gethostname(hostname, sizeof(hostname));
	he = gethostbyname(hostname);
	if (he) {
	    strncpy(hostname, he->h_name, sizeof(hostname));
	}
    }

    /* soethings gone really wrong, just hope its local only */
    if (!hostname[0]) {
	strcpy(hostname, "localhost");
    }
    s->hostname = strdup(hostname);

    if (used->ai_family == PF_INET6) {
	struct sockaddr_in6 *addr = (struct sockaddr_in6 *)used->ai_addr;

	s->port = htons(addr->sin6_port);
    } else if (used->ai_family == PF_INET) {
	struct sockaddr_in *addr = (struct sockaddr_in *)used->ai_addr;

	s->port = htons(addr->sin_port);
    } else {
	lo_throw(s, LO_UNKNOWNPROTO, "unknown protocol family", NULL);
	s->port = atoi(port);
    }

    return s;
}

void lo_server_free(lo_server s)
{
    if (s) {
	lo_method it;
	lo_method next;

	if (s->socket != -1) close(s->socket);
	freeaddrinfo(s->ai);
	free(s->hostname);
	free(s->path);
	for (it = s->first; it; it = next) {
	    next = it->next;
	    free((char *)it->path);
	    free((char *)it->typespec);
	    free(it);
	}
	free(s);
    }
}

void *lo_server_recv_raw(lo_server s, size_t *size)
{
    char buffer[LO_MAX_MSG_SIZE];
    int ret;
    void *data = NULL;

#ifdef WIN32
    if(!initWSock()) return NULL;
#endif

    s->addr_len = sizeof(s->addr);

    ret = recvfrom(s->socket, buffer, LO_MAX_MSG_SIZE, 0,
		   (struct sockaddr *)&s->addr, &s->addr_len);
    if (ret <= 0) {
	return NULL;
    }
    data = malloc(ret);
    memcpy(data, buffer, ret);

    if (size) *size = ret;

    return data;
}

void *lo_server_recv_raw_stream(lo_server s, size_t *size)
{
    struct sockaddr_storage addr;
    socklen_t addr_len = sizeof(addr);
    char buffer[LO_MAX_MSG_SIZE];
    int32_t read_size;
    int ret;
    void *data = NULL;
    int sock;

#ifdef WIN32
    fd_set ps;
    if(!initWSock()) return NULL;
    FD_ZERO(&ps);
    FD_SET(s->socket,&ps);
    if(select(1,&ps,NULL,NULL,NULL) == SOCKET_ERROR)
        return NULL;
#else
    struct pollfd ps;
    ps.fd = s->socket;
    ps.events = POLLIN | POLLPRI;
    ps.revents = 0;
    poll(&ps, 1, -1);
#endif
    sock = accept(s->socket, (struct sockaddr *)&addr, &addr_len);

    ret = recv(sock, (char *) (&read_size), sizeof(read_size), 0);
    read_size = ntohl(read_size);
    if (read_size > LO_MAX_MSG_SIZE) {
	close(sock);
	lo_throw(s, LO_TOOBIG, "Message too large", "recv()");

	return NULL;
    }
    ret = recv(sock, buffer, read_size, 0);
    //close(sock);
    if (ret <= 0) {
	return NULL;
    }
    data = malloc(ret);
    memcpy(data, buffer, ret);

    if (size) *size = ret;

    return data;
}

int lo_server_recv_noblock(lo_server s, int timeout)
{
    int sched_timeout = (int) (lo_server_next_event_delay(s) * 1000);
#ifdef WIN32
    fd_set ps;
    struct timeval stimeout;
#else
    struct pollfd ps;
#endif

#ifdef WIN32
    int res,to;

    if(!initWSock()) return 0;

    to = timeout > sched_timeout ? sched_timeout : timeout;
    stimeout.tv_sec = to/1000;
    stimeout.tv_usec = (to%1000)*1000;

    FD_ZERO(&ps);
    FD_SET(s->socket,&ps);
    res = select(1,&ps,NULL,NULL,&stimeout);

    if(res == SOCKET_ERROR)
        return 0;

    if (res || lo_server_next_event_delay(s) < 0.01)
	    return lo_server_recv(s);
#else
    ps.fd = s->socket;
    ps.events = POLLIN | POLLPRI | POLLERR | POLLHUP;
    ps.revents = 0;
    poll(&ps, 1, timeout > sched_timeout ? sched_timeout : timeout);

    if (ps.revents == POLLERR || ps.revents == POLLHUP) {
	return 0;
    }
    if (ps.revents || lo_server_next_event_delay(s) < 0.01) {
	return lo_server_recv(s);
    }
#endif

    return 0;
}

int lo_server_recv(lo_server s)
{
    void *data;
    size_t size;
    char *path;
    char *types;
    double sched_time = lo_server_next_event_delay(s);
#ifdef WIN32
    fd_set ps;
    struct timeval stimeout;
    int res;
#else
    struct pollfd ps;
#endif

again:
    if (sched_time > 0.01) {
	if (sched_time > 10.0) {
	    sched_time = 10.0;
	}

#ifdef WIN32
    if(!initWSock()) return 0;

    ps.fd_count = 1;
    ps.fd_array[0] = s->socket;

    stimeout.tv_sec = (long) sched_time;
    stimeout.tv_usec = (long) ((sched_time-stimeout.tv_sec) * 1.e6);
	res = select(1,&ps,NULL,NULL,&stimeout);
	if(res == SOCKET_ERROR) {
	    return 0;
	}

	if(!res) {
	    sched_time = lo_server_next_event_delay(s);

	    if (sched_time > 0.01) {
		goto again;
	    }

	    return dispatch_queued(s);
	}
#else
	ps.fd = s->socket;
	ps.events = POLLIN | POLLPRI | POLLERR | POLLHUP;
	ps.revents = 0;
	poll(&ps, 1, (int)(sched_time * 1000.0));

	if (ps.revents == POLLERR || ps.revents == POLLHUP) {
	    return 0;
	}

	if (!ps.revents) {
	    sched_time = lo_server_next_event_delay(s);

	    if (sched_time > 0.01) {
		goto again;
	    }

	    return dispatch_queued(s);
	}
#endif
    } else {
	return dispatch_queued(s);
    }
    if (s->protocol == LO_TCP) {
	data = lo_server_recv_raw_stream(s, &size);
    } else {
	data = lo_server_recv_raw(s, &size);
    }

    if (!data) {
	return 0;
    }
    path = data;

    types = ((char *) data) + lo_strsize(path);
    if (!strcmp(path, "#bundle")) {
	char *pos = types;
	uint32_t len;
	lo_timetag ts, now;

	lo_timetag_now(&now);

	ts.sec = lo_otoh32(*((uint32_t *)pos));
	pos += 4;
	ts.frac = lo_otoh32(*((uint32_t *)pos));
	pos += 4;

	while (pos - (char *)data < size) {
	    len = lo_otoh32(*((uint32_t *)pos));
	    pos += 4;
	    /* test for immedaite dispatch */
	    if ((ts.sec == 0 && ts.frac == 1) ||
				lo_timetag_diff(ts, now) <= 0.0) {
		types = pos + lo_strsize(pos);
		dispatch_method(s, pos, types + 1, types + lo_strsize(types));
	    } else {
		queue_data(s, ts, pos, len);
	    }
	    pos += len;
	}

	free(data);

	return size;
    } else if (*types != ',') {
	lo_throw(s, LO_ENOTYPE, "Missing typetag", path);

	return -1;
    }

    dispatch_method(s, path, types+1, data);

    free(data);

    return size;
}

/* returns the time in seconds until the next scheduled event */
double lo_server_next_event_delay(lo_server s)
{
    if (s->queued) {
	lo_timetag now;
	double delay;

	lo_timetag_now(&now);
	delay = lo_timetag_diff(((queued_msg_list *)s->queued)->ts, now);

	delay = delay > 100.0 ? 100.0 : delay;
	delay = delay < 0.0 ? 0.0 : delay;

	return delay;
    }

    return 100.0;
}

static void dispatch_method(lo_server s, const char *path, char *types,
			    void *data)
{
    int argc = strlen(types);
    lo_arg **argv = NULL;
    lo_method it;
    int ret = 1;
    int err;
    int endian_fixed = 0;
    int pattern = strpbrk(path, " #*,?[]{}") != NULL;
    lo_message msg = lo_message_new();
    lo_address src = lo_address_new(NULL, NULL);
    char hostname[LO_HOST_SIZE];
    char portname[32];
    const char *pptr;

    free(msg->types);
    msg->types = types;
    msg->typelen = strlen(types);
    msg->typesize = 0;
    msg->data = data;
    msg->datalen = 0;
    msg->datasize = 0;
    msg->source = src;

    //inet_ntop(s->addr.ss_family, &s->addr.padding, hostname, sizeof(hostname));
    if (s->protocol == LO_UDP) {
	err = getnameinfo((struct sockaddr *)&s->addr, sizeof(s->addr),
	    hostname, sizeof(hostname), portname, sizeof(portname),
	    NI_NUMERICHOST | NI_NUMERICSERV);
	if (err) {
	    switch (err) {
	    case EAI_AGAIN:
		lo_throw(s, err, "Try again", path);
		break;
	    case EAI_BADFLAGS:
		lo_throw(s, err, "Bad flags", path);
		break;
	    case EAI_FAIL:
		lo_throw(s, err, "Failed", path);
		break;
	    case EAI_FAMILY:
		lo_throw(s, err, "Cannot resolve address family", path);
		break;
	    case EAI_MEMORY:
		lo_throw(s, err, "Out of memory", path);
		break;
	    case EAI_NONAME:
		lo_throw(s, err, "Cannot resolve", path);
		break;
#ifndef WIN32
	    case EAI_SYSTEM:
		lo_throw(s, err, strerror(err), path);
		break;
#endif
	    default:
		lo_throw(s, err, "Unknown error", path);
		break;
	    }

	    return;
	}
    } else {
	hostname[0] = '\0';
	portname[0] = '\0';
    }
    
    
    // Store the source information in the lo_address
    if (src->host) free(src->host);
    if (src->host) free(src->port);
    src->host = strdup(hostname);
    src->port = strdup(portname);
    src->proto = s->protocol;

    for (it = s->first; it; it = it->next) {
	/* If paths match or handler is wildcard */
	if (!it->path || !strcmp(path, it->path) ||
	    (pattern && lo_pattern_match(it->path, path))) {
	    /* If types match or handler is wildcard */
	    if (!it->typespec || !strcmp(types, it->typespec)) {

		if (!argv && *types) {
		    int i;
		    char *ptr = types - 1 + lo_strsize(types - 1);

		    argv = calloc(argc + 1, sizeof(lo_arg *));
		    if (!endian_fixed) {
			for (i=0; i<argc; i++) {
			    argv[i] = (lo_arg *)ptr;
			    lo_arg_host_endian(types[i], ptr);
			    ptr += lo_arg_size(types[i], ptr);
			}
			endian_fixed = 1;
		    }
		}

		/* Send wildcard path to generic handler, expanded path
		  to others.
		*/
		pptr = path;
		if (it->path) pptr = it->path;
		ret = it->handler(pptr, types, argv, argc, msg,
				      it->user_data);

	    } else if (lo_can_coerce_spec(types, it->typespec)) {
		int i;
		int opsize = 0;
		char *ptr = types - 1 + lo_strsize(types - 1);
		char *data_co, *data_co_ptr;

		argv = calloc(argc+1, sizeof(lo_arg *));
		for (i=0; i<argc; i++) {
		    opsize += lo_arg_size(it->typespec[i], ptr);
		    ptr += lo_arg_size(types[i], ptr);
		}

		data_co = malloc(opsize);
		data_co_ptr = data_co;
		ptr = types - 1 + lo_strsize(types - 1);
		for (i=0; i<argc; i++) {
		    argv[i] = (lo_arg *)data_co_ptr;
		    if (!endian_fixed) {
			lo_arg_host_endian(types[i], ptr);
		    }
		    lo_coerce(it->typespec[i], (lo_arg *)data_co_ptr,
			      types[i], (lo_arg *)ptr);
		    data_co_ptr += lo_arg_size(it->typespec[i], data_co_ptr);
		    ptr += lo_arg_size(types[i], ptr);
		}
		endian_fixed = 1;

		/* Send wildcard path to generic handler, expanded path
		  to others.
		*/
		pptr = path;
		if (it->path) pptr = it->path;
		ret = it->handler(pptr, it->typespec, argv, argc, msg,
				      it->user_data);
		free(argv);
		free(data_co);
		argv = NULL;
	    }

	    if (ret == 0 && !pattern) {
		break;
	    }
	}
    }

    /* If we find no matching methods, check for protocol level stuff */
    if (ret == 1 && s->protocol == LO_UDP) {
	char *pos = strrchr(path, '/');

	/* if its a method enumeration call */
	if (pos && *(pos+1) == '\0') {
	    lo_message reply = lo_message_new();
	    int len = strlen(path);
	    lo_strlist *sl = NULL, *slit, *slnew, *slend;

	    if (!strcmp(types, "i")) {
		lo_message_add_int32(reply, argv[0]->i);
	    }
	    lo_message_add_string(reply, path);

	    for (it = s->first; it; it = it->next) {
		/* If paths match */
		if (it->path && !strncmp(path, it->path, len)) {
		    char *tmp;
		    char *sec;

		    tmp = malloc(strlen(it->path + len) + 1);
		    strcpy(tmp, it->path + len);
#ifdef WIN32
            sec = strchr(tmp,'/');
#else
		    sec = index(tmp, '/');
#endif
		    if (sec) *sec = '\0';
		    slend = sl;
		    for (slit = sl; slit; slend = slit, slit = slit->next) {
			if (!strcmp(slit->str, tmp)) {
			    free(tmp);
			    tmp = NULL;
			    break;
			}
		    }
		    if (tmp) {
			slnew = calloc(1, sizeof(lo_strlist));
			slnew->str = tmp;
			slnew->next = NULL;
			if (!slend) {
			    sl = slnew;
			} else {
			    slend->next = slnew;
			}
		    }
		}
	    }

	    for (slit = sl; slit; slit = slit->next) {
		lo_message_add_string(reply, slit->str);
		free(slit->str);
	    }
	    lo_send_message(src, "#reply", reply);
	    lo_message_free(reply);
	}
    }

    free(argv);


    /* these are already part of data and will be freed later */
    msg->data = NULL;
    msg->types = NULL;
    
    lo_address_free(src);
    lo_message_free(msg);
}

int lo_server_events_pending(lo_server s)
{
    return s->queued != 0;
}

static void queue_data(lo_server s, lo_timetag ts, void *data, size_t len)
{
    /* insert blob into future dispatch queue */
    queued_msg_list *it = s->queued;
    queued_msg_list *prev = NULL;
    queued_msg_list *ins = calloc(1, sizeof(queued_msg_list));

    ins->ts = ts;
    ins->len = len;
    ins->data = malloc(len);
    memcpy(ins->data, data, len);

    while (it) {
	if (lo_timetag_diff(it->ts, ts) > 0.0) {
	    if (prev) {
		prev->next = ins;
	    } else {
		s->queued = ins;
		ins->next = NULL;
	    }
	    ins->next = it;

	    return;
	}
	prev = it;
	it = it->next;
    }

    /* fell through, so this event is last */
    if (prev) {
	prev->next = ins;
    } else {
	s->queued = ins;
    }
    ins->next = NULL;
}

static int dispatch_queued(lo_server s)
{
    char *path, *types, *data;
    queued_msg_list *head = s->queued;
    queued_msg_list *tailhead;
    lo_timetag disp_time;

    if (!head) {
	lo_throw(s, LO_INT_ERR, "attempted to dispatch with empty queue",
		 "timeout");
	return 1;
    }

    disp_time = head->ts;

    do {
	tailhead = head->next;
	path = ((queued_msg_list *)s->queued)->data;
	types = path + lo_strsize(path) + 1;
	data = types + lo_strsize(types);
	dispatch_method(s, path, types, data);

	free(((queued_msg_list *)s->queued)->data);
	free((queued_msg_list *)s->queued);

	s->queued = tailhead;
	head = tailhead;
    } while (head && lo_timetag_diff(head->ts, disp_time) < FLT_EPSILON);

    return 0;
}

lo_method lo_server_add_method(lo_server s, const char *path,
			       const char *typespec, lo_method_handler h,
			       void *user_data)
{
    lo_method m = calloc(1, sizeof(struct _lo_method));
    lo_method it;

    if (path && strpbrk(path, " #*,?[]{}")) {
	return NULL;
    }

    if (path) {
	m->path = strdup(path);
    } else {
	m->path = NULL;
    }

    if (typespec) {
	m->typespec = strdup(typespec);
    } else {
	m->typespec = NULL;
    }

    m->handler = h;
    m->user_data = user_data;
    m->next = NULL;

    /* append the new method to the list */
    if (!s->first) {
	s->first = m;
    } else {
	/* get to the last member of the list */
	for (it=s->first; it->next; it=it->next);
	it->next = m;
    }

    return m;
}

void lo_server_del_method(lo_server s, const char *path,
			  const char *typespec)
{
    lo_method it, prev, next;
    int pattern = 0;

    if (!s->first) return;
    if (path) pattern = strpbrk(path, " #*,?[]{}") != NULL;

    it = s->first;
    prev = it;
    while (it) {
	/* incase we free it */
	next = it->next;

	/* If paths match or handler is wildcard */
	if ((it->path == path) ||
	    (path && it->path && !strcmp(path, it->path)) ||
	    (pattern && lo_pattern_match(it->path, path))) {
	    /* If types match or handler is wildcard */
	    if ((it->typespec == typespec) ||
		(typespec && it->typespec && !strcmp(typespec, it->typespec))
	        ) {
		/* Take care when removing the head. */
		if (it == s->first) {
		    s->first = it->next;
		} else {
		    prev->next = it->next;
		}
		next = it->next;
		free((void *)it->path);
		free((void *)it->typespec);
		free(it);
		it = prev;
	    }
	}
	prev = it;
	if (it) it = next;
    }
}

int lo_server_get_socket_fd(lo_server s)
{
    if (s->protocol != LO_UDP &&
        s->protocol != LO_TCP 
#ifndef WIN32
        && s->protocol != LO_UNIX
#endif
    ) {
        return -1;  /* assume it is not supported */
    }
    nyquist_printf("server_get_socket is UDP: %d\n", s->protocol == LO_UDP);
    return s->socket;
}

int lo_server_get_port(lo_server s)
{
    if (!s) {
	return 0;
    }

    return s->port;
}

char *lo_server_get_url(lo_server s)
{
    int ret;
    char *buf;

    if (!s) {
	return NULL;
    }

    if (s->protocol == LO_UDP || s->protocol == LO_TCP) {
	char *proto = s->protocol == LO_UDP ? "udp" : "tcp";

	ret = snprintf(NULL, 0, "osc.%s://%s:%d/", proto, s->hostname, s->port);
	if (ret <= 0) {
	    /* this libc is not C99 compliant, guess a size */
	    ret = 1023;
	}
	buf = malloc((ret + 2) * sizeof(char));
	snprintf(buf, ret+1, "osc.%s://%s:%d/", proto, s->hostname, s->port);

	return buf;
    } 
#ifndef WIN32
    else if (s->protocol == LO_UNIX) {
	ret = snprintf(NULL, 0, "osc.unix:///%s", s->path);
	if (ret <= 0) {
	    /* this libc is not C99 compliant, guess a size */
	    ret = 1023;
	}
	buf = malloc((ret + 2) * sizeof(char));
	snprintf(buf, ret+1, "osc.unix:///%s", s->path);

	return buf;
    }
#endif
    return NULL;
}

void lo_server_pp(lo_server s)
{
    lo_method it;

    nyquist_printf("socket: %d\n\n", s->socket);
    nyquist_printf("Methods\n");
    for (it = s->first; it; it = it->next) {
	nyquist_printf("\n");
	lo_method_pp_prefix(it, "   ");
    }
}

static int lo_can_coerce_spec(const char *a, const char *b)
{
    unsigned int i;

    if (strlen(a) != strlen(b)) {
	return 0;
    }

    for (i=0; a[i]; i++) {
	if (!lo_can_coerce(a[i], b[i])) {
	    return 0;
	}
    }

    return 1;
}

static int lo_can_coerce(char a, char b)
{
    return ((a == b) ||
           (lo_is_numerical_type(a) && lo_is_numerical_type(b)) ||
           (lo_is_string_type(a) && lo_is_string_type (b)));
}

void lo_throw(lo_server s, int errnum, const char *message, const char *path)
{
    if (s->err_h) {
	(*s->err_h)(errnum, message, path);
    }
}

/* vi:set ts=8 sts=4 sw=4: */
