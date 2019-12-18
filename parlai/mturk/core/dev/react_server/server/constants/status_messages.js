/* Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import * as agent_states from './agent_states.js'

export const DISCONNECTED = (
  'You disconnected in the middle of this HIT and were marked as inactive. ' +
  'As these HITs often require real-time interaction, it is no longer ' +
  'available for completion. Please return this HIT and accept a new one ' +
  'if you would like to try again.'
)

export const DONE = (
  'You disconnected after completing this HIT without marking it as ' +
  'completed. Please press the done button below to finish the HIT.'
)

export const EXPIRED = (
  'You disconnected in the middle of this HIT and the HIT expired before ' +
  'you reconnected. It is no longer available for completion. Please return ' +
  'this HIT and accept a new one if you would like to try again.'
)

export const PARTNER_DISCONNECT = (
  'One of your partners disconnected in the middle of the HIT. We won\'t ' +
  'penalize you for their disconnect, so please use the button below to ' +
  'mark the HIT as complete.'
)

export const RETURNED = (
  'You disconnected from this HIT and then returned it. As we have marked ' +
  'the HIT as returned, it is no longer available for completion. Please ' +
  'accept a new HIT if you would like to try again'
)

export const PARLAI_DISCONNECT = (
  'Our server went down and was unable to register completion of your task ' +
  'work. If you\'ve completed the task at hand, please reach out so that we ' +
  'can compensate for this occurrence.'
)

export const OTHER = (
  'Our server was unable to handle your reconnect properly and thus this ' +
  'HIT no longer seems available for completion. Please try to connect ' +
  'again or return this HIT and accept a new one.'
)

export function get_message_for_status(status) {
  switch (status) {
    case agent_states.STATUS_DISCONNECT:
      return DISCONNECTED;
    case agent_states.STATUS_DONE:
      return DONE;
    case agent_states.STATUS_EXPIRED:
      return EXPIRED;
    case agent_states.STATUS_PARTNER_DISCONNECT:
      return PARTNER_DISCONNECT;
    case agent_states.STATUS_RETURNED:
      return RETURNED;
    case agent_states.STATUS_PARLAI_DISCONNECT:
      return PARLAI_DISCONNECT;
    default:
      return OTHER;
  }
}
